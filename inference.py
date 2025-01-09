import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from utils.data import load_json
import torch
import psutil
import pickle
import time 
import requests
import json
from utils.model_factory import LLMGenerator
from utils.prompt_factory import PromptGenerator
from sklearn.metrics import accuracy_score


def classify_text(server, 
                  model_name, 
                  prompt_name, 
                  refine_prediction, 
                  dataset_name, 
                  dataset_metadata, 
                  test_mode, 
                  replications, 
                  temperature, 
                  max_task_tokens,
                  seed):
    """
    Classifies text using specified LLM model and prompt.

    Args:
        server (str): The server type ('transformers', 'llama.cpp', or 'openai').
        model_name (str): Name of the model to use.
        prompt_name (str): Name of the prompt to use.
        refine_prediction (boolean): Whether to refine the answer with a second prompt.
        dataset_name (str): Name of the dataset to use.
        dataset_metadata (dict): Metadata of the dataset.
        test_mode (str): 'constrained' for FSM/outlines approach, 'unconstrained' for text-generation.
        replications (int): Number of replications to perform.
        temperature (float): Temperature for text generation.
        max_task_tokens (int): Maximum number of tokens from the task to give the model.

    Returns:
        pd.DataFrame: DataFrame containing classification results.
    """

    print("Refine predictions:", refine_prediction)
    print("Test mode:", test_mode)

    # replications
    relications = list(range(0,replications))

    # placeholder
    results = []
    
    # Load the data dictionary and dataset from a pickle file
    with open(dataset_metadata["dataset_dest"], 'rb') as f:
        data_obj = pickle.load(f)
    # Dataset
    data = data_obj["dataset"]
    response_categories = data_obj["labels"]
    logging.info(f"Loaded {len(data)} tasks for classification")
    
    # create a model object
    generator = LLMGenerator(model_name, 
                            test_mode, 
                            response_categories,
                            temperature, 
                            max_response_tokens=50,
                            seed=seed)
    
    # initialization
    p_gen = PromptGenerator(dataset_name,
                            prompt_name,
                            test_mode, 
                            model_name = generator.model_attributes["model_dest"],
                            tokenizer = generator.get_tokenizer(),
                            device = "cuda") #generator.get_device())

    # tokenizer for counting tokens
    tokenizer = generator.get_tokenizer()

    # load only a tokenizer (for counting tokens)
    if server != "transformers":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Iterate over each task
    for replication in relications:

        # Generate bootstrap samples (sampling w/ replacement)
        n_samples = len(data)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        data.index = list(range(0,len(data)))
        bootstrap_sample = data.loc[indices]

        for index, row in tqdm(bootstrap_sample.iterrows(), total=bootstrap_sample.shape[0], desc=f"Classifying with {model_name} and {prompt_name}"):
            # Load the task from the table
            task_id = row["TaskID"]
            text = row["Text"]
            label = row["Label"]

            # Initialize process for memory logging
            process = psutil.Process()

            ### constrain to only n tokens
            # Tokenize the text
            tokens = tokenizer.tokenize(text)
            
            # Truncate the tokens to the first 4000 tokens
            truncated_tokens = tokens[:max_task_tokens]

            # Convert the truncated tokens back to a string
            truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)

            # Generate and encode the prompt based on the truncated text
            encoded_prompt = p_gen.create_and_encode_prompt(truncated_text)

            # token counting
            n_tokens_raw = len(tokens)
            n_tokens_trunc = len(truncated_tokens)

            # Generate prediction using the model
            start_time = time.time()
            
            # using outlines FSM constrained output
            if test_mode == "constrained":                    
                prediction = generator.constrained_generator(encoded_prompt)
            if test_mode == "unconstrained":
                if refine_prediction == "False":
                    prediction = generator.unconstrained_generator(encoded_prompt)
                if refine_prediction == "True":
                    """Chain-of-thought building blocks"""
                    # get the first model response
                    response_1 = generator.unconstrained_generator(encoded_prompt)
                    # feed the first model response back, while specifying to use the refine prompt
                    encoded_prompt = p_gen.create_and_encode_prompt(response_1, refine=True)
                    # generate the final prediction
                    prediction = generator.unconstrained_generator(encoded_prompt)


            # Track RAM usage
            ram_usage = process.memory_info().rss / 1024 / 1024 / 1024 # Convert to MB

            # Track VRAM usage (if CUDA is available)
            if torch.cuda.is_available():
                vram_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 # Convert to MB

            end_time = time.time()

            if server == "transformers":
                # For constrained generation no encoding is needed 
                if test_mode == "unconstrained":
                    #decode with tokenizer
                    prompt_text = tokenizer.convert_tokens_to_string(encoded_prompt)
            
                # For constrained generation no encoding is needed 
                if test_mode == "constrained":
                    prompt_text = encoded_prompt
            
            if server == "llama.cpp":
                #llama.cpp is not encoded
                prompt_text = str(encoded_prompt)

            # Append the result to the results list
            results.append({
                'replication':replication,
                'server':generator.model_attributes["server"],
                'model': model_name,
                'model_type':generator.model_attributes["model_type"],
                'chat_format':generator.model_attributes["chat_format"],
                'prompt': prompt_name,
                'dataset': dataset_metadata["dataset_dest"],
                'task_id': task_id,
                'duration_sec':end_time - start_time,
                'vram_used_gb':vram_usage,
                'ram_used_gb':ram_usage,
                "raw_tokens":n_tokens_raw,
                "n_trunc_tokens":n_tokens_trunc,
                'input_text': truncated_text,
                'prompt_text': prompt_text,
                'predicted_label': prediction,
                'actual_label': label
            })
            logging.debug(f"ID {task_id}: Predicted: '{prediction}', Actual: '{label}'")
            
    if server in ["transformers", "llama.cpp"]:
        # delete the model from memory
        generator.cleanup()
        
        del generator
        del p_gen
        del tokens
        del truncated_tokens

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        logging.info(f"Unloaded model and cleared GPU memory")

    # store as dataframe
    results = pd.DataFrame(results)
    acc = accuracy_score( results["actual_label"], results["predicted_label"])
    logging.info("Accuracy: "+str(acc))
    
    # Convert results to a DataFrame
    return results