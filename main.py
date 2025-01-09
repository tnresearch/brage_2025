import pandas as pd
import numpy as np
import argparse
import itertools
from utils.data import load_json
from inference import classify_text
from sklearn.metrics import accuracy_score
import os, shutil
from datetime import datetime
from codecarbon import EmissionsTracker
import torch
import gc
import os
import sys
import h5py
import logging

# Set up logging
if os.path.exists("latest_run.log"):
    os.remove("latest_run.log")

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler("latest_run.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logging.getLogger().setLevel(logging.DEBUG)

def get_active_conda_env():
    """
    Gets the active conda env as a string
    """
    return os.path.basename(sys.prefix)

def validate_inputs(input_models, input_datasets, model_config, dataset_config):
    """
    Validates input models and datasets against configurations.

    Args:
        input_models (list): List of input model names.
        input_datasets (list): List of input dataset names.
        model_config (dict): Model configuration dictionary.
        dataset_config (dict): Dataset configuration dictionary.

    Returns:
        tuple: Lists of valid input models and datasets.
    """
    # Validate models
    valid_model_dests = [model_info['model_dest'] for model_info in model_config.values()]
    valid_input_models = [model for model in input_models if model in valid_model_dests]
    invalid_models = set(input_models) - set(valid_input_models)
    
    if invalid_models:
        logging.warning(f"The following models are not in models.json and will be ignored: {', '.join(invalid_models)}")
        logging.warning(f"Valid models: {', '.join(valid_model_dests)}")
    
    # Validate datasets
    valid_dataset_names = [dataset_info['dataset_name'] for dataset_info in dataset_config.values()]
    valid_input_datasets = [dataset for dataset in input_datasets if dataset in valid_dataset_names]
    invalid_datasets = set(input_datasets) - set(valid_input_datasets)
    
    if invalid_datasets:
        logging.warning(f"The following datasets are not in datasets.json and will be ignored: {', '.join(invalid_datasets)}")
        logging.warning(f"Valid datasets: {', '.join(valid_dataset_names)}")
    
    return valid_input_models, valid_input_datasets

def main(server, 
         models, 
         datasets, 
         replications=10, 
         test_modes=["unconstrained"], 
         temperatures=[0.75], 
         max_task_tokens=[2500], 
         prompts=None, 
         suffix="",
         refine=[False],
         seed=True): 
    """
    Main function to run the LLM benchmark framework.

    Args:
        server (str): The server type to use.
        models (list): List of models to evaluate.
        datasets (list): List of datasets to use.
        replications (int): Number of replications for each run.
        test_modes (list): List of test modes to use.
        temperatures (list): List of temperature values to use.
        max_task_tokens (list): List of maximum task token values to use.
        prompts (list): List of one or more prompts to used specifically (if not specified, all relevant prompts are used).
        refine (list): Whether to refine the answer with a second prompt.
    """
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the classification process")

    # Create cache directory if it doesn't exist
    cache_dir = "models/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Make a dir for all results
    if not os.path.exists("results/"):
        os.makedirs("results/")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = "results/"+current_datetime+"_"+suffix
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    # Copy current configurations into experiment folder
    files = ["models.json", "prompts.json","datasets.json"]
    for file in files:
        shutil.copyfile("configs/"+file, current_dir+"/"+file)
    # Load configurations
    prompt_config = load_json('configs/prompts.json')
    dataset_config = load_json('configs/datasets.json')['datasets']

    # Function to filter prompts based on selected datasets and specified prompts
    def filter_prompts(prompt_config, datasets, specified_prompts=None):
        filtered_prompts = {}
        for dataset, prompts in prompt_config.items():
            if dataset in datasets:
                if specified_prompts:
                    filtered_prompts[dataset] = {k: v for k, v in prompts.items() if k in specified_prompts}
                else:
                    filtered_prompts[dataset] = prompts
        return filtered_prompts

    # Filter prompts based on selected datasets and specified prompts
    filtered_prompt_config = filter_prompts(prompt_config, datasets, prompts)

    # Get all prompt names from the filtered prompt config
    prompts = set()
    for dataset_prompts in filtered_prompt_config.values():
        prompts.update(dataset_prompts.keys())

    # Define the design table
    factor_dict = {
        'models': models,
        'prompts': list(prompts),
        'refine':refine,
        'data': datasets,
        'replications': [replications],
        'test_mode': test_modes,
        'temperature': temperatures,
        'max_task_tokens': max_task_tokens
    }

    # Generate full factorial design
    design = list(itertools.product(*factor_dict.values()))
    design_df = pd.DataFrame(design, columns=factor_dict.keys())
    design_df["balanced_design"] = True
    orig_len = len(design_df)

    # Copy orig design;
    orig_design = design_df.copy()

    # Filter out prompts with datasets not in datasets.json
    valid_datasets = set(dataset_config.keys())
    valid_prompts = [prompt for dataset, dataset_prompts in filtered_prompt_config.items() 
                    for prompt, prompt_info in dataset_prompts.items() 
                    if dataset in valid_datasets]

    # Filter design table to include only valid prompts
    design_df = design_df[design_df['prompts'].isin(valid_prompts)]
    design_df = design_df.reset_index(drop=True)
    new_len = len(design_df)

    if orig_len != new_len:
        logging.warning("Design table is **unbalanced** due to mismatch between prompt and model pairs (instruct vs. base). Please remember this when cross-tabulating the results.")
        logging.warning(f"Original num. runs: '{orig_len}'")
        logging.warning(f"New num. runs: '{new_len}'")
        design_df["balanced_design"] = False
    ###########################################################

    # add run number to table
    design_df["run"] = range(0,len(design_df))

    # print experimental design
    print("#"*25+"\n## Experimental design ##\n"+"#"*25+"\n")
    print(design_df)

    # Store design table
    design_df.to_csv(current_dir+"/design_table.csv",index=False)
    orig_design.to_csv(current_dir+"/unfiltered_design_table.csv",index=False) # Original design table 
    logging.info(f"Loaded config files, and created experiment table of '{len(design_df)}' runs. This can be found in results/design_table.csv")


    # Placeholders
    results_list = []
    agg_metrics_list = []

    # Perform classification tasks for each combination of data, models and prompts
    for index, row in design_df.iterrows():
        
        # define variables of importance for the run
        model_name = row["models"]
        prompt_name = row["prompts"]
        refine = row["refine"]
        dataset_name = row["data"]
        replications = row["replications"]
        temperature = row["temperature"]
        test_mode = row["test_mode"]
        max_task_tokens = row["max_task_tokens"]
        dataset_metadata = dataset_config[dataset_name]
        run = row["run"]

        # Create an emissions tracker with the desired output directory
        tracker = EmissionsTracker(output_dir=current_dir, project_name="Benchmark", experiment_id=run, log_level="error")

        logging.info(f"Classifying with'{server}', model '{model_name}' and prompt '{prompt_name}' on dataset '{dataset_name}'")
        tracker.start()
        predictions_df = classify_text(server, model_name, prompt_name, refine, dataset_name, dataset_metadata, test_mode, replications, temperature, max_task_tokens, seed)
        predictions_df["RUN"] = run
        tracker.stop()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()        
        gc.collect()
        logging.info(f"Done with model '{model_name}' and prompt '{prompt_name}' on dataset '{dataset_name}'")

        """
        The following
        """

        # placeholder
        accuracies = []
        durations = []
        reps = []
        vram = []

        # iterate over the replications to create vectors
        for rep in predictions_df.replication.unique():
            # subset on replication i
            sub = predictions_df.loc[predictions_df["replication"] == rep]
            
            # calculate metrics over the replication i
            predicted_label = sub['predicted_label']
            actual_label = sub['actual_label'] 

            vram_used = sub["vram_used_gb"]

            accuracies.append(accuracy_score(actual_label, predicted_label))
            durations.append(np.mean(sub["duration_sec"]))
            reps.append(rep+1)
            vram.append(np.mean(vram_used))

        # Generate the design table subset on row i
        row_df = pd.DataFrame(row).T
        row_df.columns = design_df.columns

        # generate replication level table

        # Repeat this row for every replication
        agg_metrics_df = pd.concat([row_df]*replications, ignore_index=True)
        agg_metrics_df["replication"] = reps
        agg_metrics_df["accuracy"] = accuracies
        agg_metrics_df["avg_vram_used"] = vram
        agg_metrics_df["avg_duration_sec"] = durations
        
        # append the results
        results_list.append(predictions_df)
        agg_metrics_list.append(agg_metrics_df)

        # Concat to DF (saving after each run)
        predictions_df = pd.concat(results_list, ignore_index=True)
        agg_metrics = pd.concat(agg_metrics_list, ignore_index=True)
        
        # Format the tables
        ####################
        """
        predictions_df['run'] = predictions_df['run'].astype(int)
        predictions_df['replication'] = predictions_df['replication'].astype(int)
        predictions_df['server'] = predictions_df['server'].astype('category')
        predictions_df['models'] = predictions_df['models'].astype('category')
        predictions_df['model_type'] = predictions_df['model_type'].astype('category')
        predictions_df['chat_format'] = predictions_df['chat_format'].astype('category')
        predictions_df['prompts'] = predictions_df['prompts'].astype('category')
        predictions_df['data'] = predictions_df['data'].astype('category')
        predictions_df['test_mode'] = predictions_df['test_mode'].astype('category')
        predictions_df['balanced_design'] = predictions_df['balanced_design'].astype(bool)
        predictions_df['temperature'] = predictions_df['temperature'].astype(float)
        predictions_df['max_task_tokens'] = predictions_df['max_task_tokens'].astype(int)
        """

        ####################
        
        # Save results to hdf5
        predictions_df.to_hdf(current_dir+'/predictions.h5', key='predictions', mode='w', format="table")

        # Format output table
        agg_metrics['run'] = agg_metrics['run'].astype(str)
        agg_metrics['replications'] = agg_metrics['replications'].astype(str)
        agg_metrics['models'] = agg_metrics['models'].astype('category')
        agg_metrics['prompts'] = agg_metrics['prompts'].astype('category')
        agg_metrics['data'] = agg_metrics['data'].astype('category')
        agg_metrics['test_mode'] = agg_metrics['test_mode'].astype('category')
        agg_metrics['balanced_design'] = agg_metrics['balanced_design'].astype(bool)
        agg_metrics['temperature'] = agg_metrics['temperature'].astype(float)
        agg_metrics['max_task_tokens'] = agg_metrics['max_task_tokens'].astype(str)
        agg_metrics['avg_vram_used'] = agg_metrics['avg_vram_used'].astype(float)
        

        agg_metrics.to_hdf(current_dir+'/agg_metrics.h5', key='agg_metrics', mode='w', format="table")

        logging.info(f"Done with dataset: '{dataset_name}', results saved.")

    if orig_len != new_len:
        logging.warning("Design table is **unbalanced** due to mismatch between prompt and model pairs (instruct vs. base). Please remember this when cross-tabulating the results.")
        logging.warning(f"Original num. runs: '{orig_len}'")
        logging.warning(f"New num. runs: '{new_len}'")
    
    logging.info("Classification completed. Results saved to "+current_dir+"/ folder")
    shutil.copyfile("latest_run.log", current_dir+"/"+"logfile.log")

    #results_message = ("\n\nResults:" + "\nMax achieved accuracy in experiments: "+str(np.max(agg_metrics_df["accuracy"]))+"\n" + "Average inference time in secs: "+ str(np.mean(agg_metrics_df["avg_duration_sec"])))
        
    #logging.info(results_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation framework")
    
    # Remove the --server argument as this is the conda env
    # parser.add_argument("--server", required=True, help="Whether to use transformers, llama.cpp or other api")
    
    parser.add_argument("--models", nargs='+', required=True, help="List of model_dest values to evaluate")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of dataset names to evaluate")
    parser.add_argument("--replications", type=int, default=10, help="Number of replications")
    parser.add_argument("--test_modes", nargs='+', default=["unconstrained"], help="Test modes")
    parser.add_argument("--temperatures", nargs='+', type=float, default=[0.75], help="Temperature values")
    parser.add_argument("--max_task_tokens", nargs='+', type=int, default=[2500], help="Maximum task tokens")
    parser.add_argument("--prompts", nargs='+', default=None, help="List of specific prompt names to evaluate")
    parser.add_argument("--suffix",  type=str, default="", help="Suffix to add to the results directory")
    parser.add_argument("--refine", nargs='+', default=["False"], help="List of strings denoting whether or not to refine the answer in unconstrained mode")
    parser.add_argument("--seed", type=bool, default=True, help="Whether to set a seed or not (seed is set to 1337).")

    args = parser.parse_args()
    
    # Get the active Conda environment name
    server = get_active_conda_env()
    
    
    # Load configurations
    model_config = load_json('configs/models.json')['models']
    dataset_config = load_json('configs/datasets.json')['datasets']
    
    # Validate and filter input models and datasets
    validated_models, validated_datasets = validate_inputs(args.models, args.datasets, model_config, dataset_config)
    
    if not validated_models or not validated_datasets:
        logging.error("No valid models or datasets specified. Exiting.")
        exit(1)
    
    main(server, # conda env
         validated_models, 
         validated_datasets, 
         args.replications, 
         args.test_modes, 
         args.temperatures, 
         args.max_task_tokens, 
         args.prompts, 
         args.suffix,
         args.refine,
         args.seed)