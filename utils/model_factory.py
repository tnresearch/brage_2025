import pandas as pd
from tqdm import tqdm
import logging
import torch
import time
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
import requests
import json
from utils.data import download_and_cache_model

class LLMGenerator:
    """
    This class creates the LLM object using the appropriate specifications, so that it can be re-used.
    """
    def __init__(self, 
                 model_name: str, 
                 test_mode: str, 
                 response_categories: list, 
                 temperature: float, 
                 max_response_tokens: int,
                 seed: bool):
        
        self.model_name = model_name
        self.cache_dir = "models/"
        self.test_mode = test_mode
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self.response_categories = response_categories
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.seed = seed

        self.uncon_gen_config = GenerationConfig(
                                max_new_tokens=self.max_response_tokens,
                                do_sample = True, # Whether or not to use sampling
                                temperature=self.temperature,
                                num_beams = 2, #  Number of beams for beam search. 1 means no beam search.
                                max_time = 10.0, # max used time in sec to generate response with search methods
                                top_k = 50, # The number of highest probability vocabulary tokens to keep for top-k-filtering.
                                top_p = 0.75, #default=1.0, # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                                #repetition_penalty = 1, # The parameter for repetition penalty.
                                #length_penalty = 1, # Exponential penalty to the length that is used with beam-based generation.
                                #pad_token_id (int, optional) — The id of the padding token.
                                #bos_token_id (int, optional) — The id of the beginning-of-sequence token.
                                #eos_token_id (Union[int, List[int]], optional) — The id of the end-of-sequence token. Optionally, use a list to set multiple end-of-sequence tokens.
                            )

        # load model attributes based on model name
        from utils.data import load_json
        self.model_attributes = load_json('configs/models.json')["models"][self.model_name]   

        self.server = self.model_attributes["server"]     

        self.create_generator()

    def create_generator(self):
        """Loads a model based on the conditions."""
        if self.server == "transformers":
            torch.cuda.empty_cache()

            # Set a seed:
            if self.seed == True:
                set_seed(1337)
                logging.info("NOTE: Seed set in transformers library to 1337")
            if self.seed == False:
                logging.info("NOTE: No seed is set, results are pseudo-random")

            # Use the new download_and_cache_model function
            cached_model_path = download_and_cache_model(self.model_name, self.cache_dir)

            self.llm = AutoModelForCausalLM.from_pretrained(cached_model_path, torch_dtype=torch.bfloat16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(cached_model_path)
            
            logging.info(f"Loaded LLM from cache: {cached_model_path}")
            #logging.info(f"Loaded LLM: {self.model_name}")
            logging.info(f"Test mode: {self.test_mode}")
            
            if self.test_mode == "constrained":
                # import modules
                import outlines
                from outlines import models

                # convert to outlines format
                model = outlines.models.Transformers(self.llm, self.tokenizer)
                torch.cuda.empty_cache()

                # Outlines constrained output generator
                self.generator = outlines.generate.choice(model, self.response_categories)
                logging.info(f"Constrained LLM to: {self.response_categories}")

                #clean up
                del model
                del self.llm
                torch.cuda.empty_cache()

            if self.test_mode == "unconstrained":                
                #clean up
                self.model = self.llm

                # set model generation parameters
                print("Modifying default generation config")
                self.model.generation_config.do_sample = True, # Whether or not to use sampling
                self.model.generation_config.num_beams = 2, #  Number of beams for beam search. 1 means no beam search.
                self.model.generation_config.temperature=self.temperature,
                self.model.generation_config.top_k = 50, # The number of highest probability vocabulary tokens to keep for top-k-filtering.
                self.model.generation_config.top_p = 0.75, #default=1.0, # If set to float < 1, only the smallest set of most probable tokens 
                self.model.generation_config.max_new_tokens=self.max_time = 10.0, # max used time in sec to generate response with search methods

                # create the generator
                self.generator = self.unconstrained_generator
                
                del self.llm
                torch.cuda.empty_cache()

        if self.server == "openai":
            if self.test_mode == "constrained":
                assert True, "Error: OpenAI and Outlines not compatible"
            if self.test_mode == "uconstrained":
                self.generator = self.chat_with_openai

        if self.server == "llama.cpp":
            # Import modules
            from llama_cpp import Llama
            
            # Create model object
            self.llm = Llama(model_path=self.model_name, 
                                n_ctx=4096, # Context size (max tokens) for the model. Default is 2048.
                                n_threads=12,
                                n_gpu_layers=-1, # Number of layers to offload to the GPU. Default is -1, which attempts to offload all layers.
                                n_batch=4096,  # Maximum batch size for prompt processing. Default is 512.
                                seed=-1, # Random seed for generation. Default is -1.
                                verbose=True, # Whether to print verbose output. Default is False.
                                #structured_output={}, #Configuration for generating structured output. Can be a dictionary or a more detailed configuration object.
                                chat_format=self.model_attributes["chat_format"])
            
            # tokenizer if any
            self.tokenizer = self.llm.tokenizer()
            logging.info(f"Loaded LLM: {self.model_name} using llama.cpp")


    def constrained_generator(self, inputs):
        if self.server == "transformers":
            response = self.generator(inputs)

        if self.server == "llama.cpp":
            # Import modules
            from llama_cpp import Llama, LlamaGrammar

            # prompt
            input_text = inputs
            
            # Create the GBNF grammar string            
            def create_gbnf_grammar(valid_values):
                # Escape any double quotes in the valid values
                escaped_values = [value.replace('"', '\\"') for value in valid_values]
                
                # Create the grammar string
                grammar_string = 'root ::= ' + ' | '.join(f'"{value}"' for value in escaped_values)
                
                return grammar_string
            
            # create the gramar string
            grammar_string = create_gbnf_grammar(self.response_categories)

            # Compile the grammar
            grammar = LlamaGrammar.from_string(grammar_string)

            output = self.llm.create_chat_completion(input_text, 
                              max_tokens=self.max_response_tokens,
                              temperature=self.temperature, # Controls randomness in generation. Higher values increase randomness. Default is 1.0.
                              frequency_penalty = 0.0, # Penalty for repeated tokens. Default is 0.0.
                              presence_penalty = 0.0, # Penalty for new tokens. Default is 0.0.
                              top_k=50,
                              top_p=0.75, #Cumulative probability threshold for token sampling. Default is 1.0.
                              grammar=grammar  # Pass the compiled grammar
            )
            
            full_output = output['choices'][0]['message']['content']
            response = full_output
        return response

    def unconstrained_generator(self, inputs):
        if self.server == "transformers":
            
            # Normistral adaptation due to error on token type ids;
            norallm_models = ["norallm/normistral-7b-warm-instruct", "norallm/normistral-7b-warm"]
            if self.model_name in norallm_models:
                inputs.pop('token_type_ids', None)
            
            # Convert input_ids back to a string
            input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

            # Documentation:
            # https://huggingface.co/docs/transformers/main_classes/text_generation
            outputs = self.model.generate(**inputs, 
                                          generation_config=self.uncon_gen_config
                                          )

            # Decode the full output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check if input_text is in full_output and remove it
            if input_text in full_output:
                response = full_output[full_output.index(input_text) + len(input_text):].strip()
                #response = full_output[len(input_text):].strip()
            else:
                # If input_text is not in full_output, log a warning and return the full output
                logging.warning(f"Input text not found in model output. Returning full output.")
                response = full_output.strip()

        if self.server == "llama.cpp":
            # Import modules
            from llama_cpp import Llama

            input_text = inputs
            
            output = self.llm.create_chat_completion(input_text, 
                              max_tokens=self.max_response_tokens,
                              temperature=self.temperature, # Controls randomness in generation. Higher values increase randomness. Default is 1.0.
                              frequency_penalty = 0.0, # Penalty for repeated tokens. Default is 0.0.
                              presence_penalty = 0.0, # Penalty for new tokens. Default is 0.0.
                              top_k=50,
                              top_p=0.75 #Cumulative probability threshold for token sampling. Default is 1.0.
                              )
            
            full_output = output['choices'][0]['message']['content']
            response = full_output
        return response

    def get_tokenizer(self):
        """Getter method to retrieve the tokenizer"""
        return self.tokenizer

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'generator'):
            del self.generator
        if hasattr(self, 'llm') and self.server == "llama.cpp":
            del self.llm
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logging.info("LLMGenerator cleaned up")

    def generate(self, prompt):
        """Method to generate response using the appropriate generator"""
        return self.generator(prompt)
    
    def chat_with_openai(self, prompt):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer API_KEY" #org. API key
        }
        data = {
            "model":"gpt-4", #fixed for now
            "messages": prompt, #chatml format
            "temperature": self.temperature,
            "max_tokens":self.max_response_tokens,
            "top_p":1.0,
        }

        response = requests.post(url, headers=headers, json=data)
        assistant_reply = response['choices'][0]['message']['content']
        return assistant_reply
    
