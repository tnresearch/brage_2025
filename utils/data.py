import json
import logging
import os
from huggingface_hub import snapshot_download

def load_json(file_path):
    """
    Load a JSON file and return its content.
    
    Parameters:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded data from {file_path}")
    return data

def load_models(model_config):
    """
    Load models based on the provided configuration.
    
    Parameters:
        model_config (dict): Dictionary where keys are model names and values are model paths.
        
    Returns:
        dict: Dictionary of loaded models.
    """
    from outlines import models
    loaded_models = {}
    for name, path in model_config.items():
        loaded_models[name] = models.transformers(path, device="cuda")
        logging.info(f"Loaded model {name} from {path}")
    return loaded_models

def download_and_cache_model(model_name, cache_dir):
    """
    Downloads and caches a model if it doesn't exist locally.
    
    Args:
    model_name (str): The name or path of the model to download.
    cache_dir (str): The directory to cache the model files.
    
    Returns:
    str: The path to the cached model.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract only the model name without the organization
    model_folder_name = model_name.split('/')[-1]
    cached_model_path = os.path.join(cache_dir, model_folder_name)
    
    if not os.path.exists(cached_model_path):
        print(f"Downloading and caching model: {model_name}")
        snapshot_download(
            repo_id=model_name,
            local_dir=cached_model_path,
            allow_patterns=["*.md", "*.safetensors", "*.json", "*.txt", "tokenizer.model"],
            ignore_patterns=["*.ot", "*.msgpack", "optimizer.pt", "*.bin", "tf_model.h5"],
        )
    
    return cached_model_path