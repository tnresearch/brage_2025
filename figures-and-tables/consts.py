COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
}

colors = {
    COLORS["orange"]: [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
    ],
    COLORS["blue"]: [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
    ],
    COLORS["green"]: [
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ],
    COLORS["pink"]: [
        "norallm/normistral-7b-scratch",
        "norallm/normistral-7b-warm",
        "norallm/normistral-7b-warm-instruct",
    ],
    COLORS["purple"]: [
        "NorwAI/NorwAI-Mistral-7B",
        "NorwAI/NorwAI-Mistral-7B-instruct",
    ],
    COLORS["red"]: [
        "RuterNorway/Llama-2-7b-chat-norwegian",
        "RuterNorway/Llama-2-13b-chat-norwegian",
    ],
    COLORS["brown"]: [
        "bineric/NorskGPT-Llama3-8b",
        "bineric/NorskGPT-Mistral-7b",
    ],
}
model_to_color = {model: color for color, models in colors.items() for model in models}

model_map = {
    "google/gemma-2-2b-it": "Gemma2 2B IT",
    "google/gemma-2-9b-it": "Gemma2 9B IT",
    "google/gemma-2-27b-it": "Gemma2 27B IT",
    "google/gemma-2-2b": "Gemma2 2B",
    "google/gemma-2-9b": "Gemma2 9B",
    "google/gemma-2-27b": "Gemma2 27B",
    "meta-llama/Meta-Llama-3-8B": "Llama3 8B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama3 8B IT",
    "meta-llama/Meta-Llama-3.1-8B": "Llama3.1 8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama3.1 8B IT",
    "meta-llama/Llama-2-7b-chat-hf": "Llama2 7B Chat",
    "meta-llama/Llama-2-13b-chat-hf": "Llama2 13B Chat",
    "RuterNorway/Llama-2-7b-chat-norwegian": "Llama2 7B Chat-Nor",
    "RuterNorway/Llama-2-13b-chat-norwegian": "Llama2 13B Chat-Nor",
    "norallm/normistral-7b-scratch": "Normistral 7B Scratch",
    "norallm/normistral-7b-warm": "Normistral 7B Warm",
    "norallm/normistral-7b-warm-instruct": "Normistral 7B Warm IT",
    "bineric/NorskGPT-Llama3-8b": "NorskGPT Llama 3 8B",
    "bineric/NorskGPT-Mistral-7b": "NorskGPT Mistral 7B",
    "mistralai/Mistral-7B-v0.1": "Mistral 7B v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7B v0.1 IT",
    "NorwAI/NorwAI-Mistral-7B": "NorwAI Mistral 7B",
    "NorwAI/NorwAI-Mistral-7B-instruct": "NorwAI Mistral 7B IT",
}


model_tune_map = {
    "google/gemma-2-2b-it": "IT",
    "google/gemma-2-9b-it": "IT",
    "google/gemma-2-27b-it": "IT",
    "google/gemma-2-2b": "P",
    "google/gemma-2-9b": "P",
    "google/gemma-2-27b": "P",
    "meta-llama/Llama-2-7b-chat-hf": "IT",
    "meta-llama/Llama-2-13b-chat-hf": "IT",
    "meta-llama/Meta-Llama-3-8B": "P",
    "meta-llama/Meta-Llama-3-8B-Instruct": "IT",
    "meta-llama/Meta-Llama-3.1-8B": "P",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "IT",
    "RuterNorway/Llama-2-7b-chat-norwegian": "IT + FNB",
    "RuterNorway/Llama-2-13b-chat-norwegian": "IT + FNB",
    "norallm/normistral-7b-scratch": "PNB",
    "norallm/normistral-7b-warm": "PNB",
    "norallm/normistral-7b-warm-instruct": "PNB + FNB",
    "bineric/NorskGPT-Llama3-8b": "IT + FNB",
    "bineric/NorskGPT-Mistral-7b": "IT + FNB",
    "mistralai/Mistral-7B-v0.1": "P",
    "mistralai/Mistral-7B-Instruct-v0.1": "IT",
    "NorwAI/NorwAI-Mistral-7B": "PNB",
    "NorwAI/NorwAI-Mistral-7B-instruct": "PNB + FNB",
}


scandeval_categories_nynorsk = {
    "entity_recognition": ["norne_nn"],
    "grammaticality": ["scala_nn"],
}
scandeval_categories = {
    "entity_recognition": "NorNE-nb",
    "sentiment_analysis": "NoReC",
    # "summarization": "no_sammendrag",
    # "summarization": "sammendrag",
    # "grammaticality": "scala_nb",
    "question_answering": "NorQuAD",
    # "language_understanding": "mmlu_no",
    "commonsense_reasoning": "HellaSwag",
}

dataset_name_map = {
    "BRAGE": "BRAGE",
    "NoReC": "NoReC",
    "NorNE-nb": "NorNE-nb",
    "brage_full_desc": "BRAGE (full desc.)",
    "brage_keywords": "BRAGE (keywords)",
    "NorQuAD": "NorQuAD",
    "HellaSwag": "HellaSwag",
}
