import pandas as pd
from tqdm import tqdm
import logging
import torch
import time
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
from utils.data import load_json

class PromptGenerator:
    """
    This class generates a prompt template using model and prompt specs, and is re-used for encoding various tasks. 
    This is expected to be re-initialized between models, prompts and datasets.
    """

    def __init__(self, dataset_name: str, prompt_name: str, test_mode: str, model_name: str,
                 tokenizer: object, device: object):
        # Initialize attributes
        self.dataset_name = dataset_name
        self.prompt_name = prompt_name
        self.test_mode = test_mode
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.prev_prompt = ""

        # Load model attributes based on model name
        self.model_attributes = load_json('configs/models.json')["models"][self.model_name]
        
        # Load prompt attributes based on prompt name
        self.prompt_attributes = load_json('configs/prompts.json')[self.dataset_name][self.prompt_name]

        logging.info(f"Loaded prompt attributes: {self.prompt_name}")

    def create_and_encode_prompt(self, task, refine=False):
        """Creates the prompt, based on the settings needed"""
        
        sysprompt = self._get_sysprompt(refine)
        
        # Dictionary mapping server types to their respective handler methods
        server_handlers = {
            "transformers": self._handle_transformers,
            "llama.cpp": self._handle_llama_cpp,
            "openai": self._handle_openai
        }
        
        # Get the appropriate handler based on the server type
        handler = server_handlers.get(self.model_attributes["server"], lambda *args: None)
        inputs = handler(sysprompt, task)
        
        # Update the previous prompt
        self.prev_prompt = sysprompt
        return inputs
    
    ###################################################################
    # These are helper functions needed to run create_and_encode_prompt
    def _get_sysprompt(self, refine):
        """Get the system prompt based on whether it's a refinement or not"""
        if refine == True:
            # use the refine-prompt, assuming its specified
            prompt = self.prompt_attributes["refine_prompt"]["inst_1"]
        if refine == False:
            # return only the sysprompt
            prompt = self.prompt_attributes["sysprompt"]
        return prompt

    def _handle_transformers(self, sysprompt, task):
        """Handle prompts for transformers-based models"""
        if self.test_mode == "constrained":
            return self._handle_constrained_transformers(sysprompt, task)
        elif self.test_mode == "unconstrained":
            return self._handle_unconstrained_transformers(sysprompt, task)

    def _handle_constrained_transformers(self, sysprompt, task):
        """Handle constrained generation for transformers"""
        conversation = self._create_conversation(sysprompt, task)
        return self._format_prompt(conversation)

    def _handle_unconstrained_transformers(self, sysprompt, task):
        """Handle unconstrained generation for transformers"""
        if self.prompt_attributes["prompt_type"] == "zero-shot":
            conversation = self._create_conversation(sysprompt, task)
            formatted_prompt = self._format_prompt(conversation)
            return self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

    def _handle_llama_cpp(self, sysprompt, task):
        """Handle prompts for llama.cpp-based models"""
        if self.prompt_attributes["prompt_type"] == "zero-shot":
            conversation = self._create_conversation(sysprompt, task)
            return self._format_prompt(conversation)

    def _handle_openai(self, sysprompt, task):
        """Handle prompts for OpenAI API"""
        # Placeholder for OpenAI API handling
        return ""

    ##############################################################
    # These are the building blocks for transformers and llama.cpp
    def _create_conversation(self, sysprompt, task):
        """Creates the conversation structure based on the instruction role"""

        if self.model_attributes["instruction_role"] == "system":
            conversation = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": task},
            ]
        elif self.model_attributes["instruction_role"] == "user":
            conversation = [
                {"role": "user", "content": sysprompt + task},
            ]
        return conversation
 
    def _format_prompt(self, conversation):
        """Formats the prompt based on the chat format"""
        # Get the chat format
        chat_format = self.model_attributes["chat_format"]
        # Conditional formatting
        if chat_format == "chatml":
            # ChatML has auto prompt formatting (newer models)
            prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        elif chat_format == "alpaca":
            # Alpaca models need manual formatting
            prompt = f"### Instruction:\n{conversation[0]['content']}\n### Input:\n{conversation[1]['content']}\n### Response:\n"
        elif chat_format == "unknown":
            # For models that are not trained with Alpaca or ChatML
            prompt = conversation[0]['content'] + conversation[1]['content']
        return prompt