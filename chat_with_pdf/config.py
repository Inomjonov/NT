"""Settings shared by the whole app."""
import torch

# Hugging Face models, downloaded automatically on first use
LLM_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",  # ~3 GB, fast
    "Qwen/Qwen2.5-3B-Instruct",  # ~6 GB, follows instructions better
]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Splitting pages into chunks (sizes are in characters)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Agent
TOP_K = 4  # passages returned by one search
MAX_REWRITES = 1  # how many times the agent may retry a search that found nothing useful
MAX_NEW_TOKENS = 512
MAX_HISTORY_MESSAGES = 6  # earlier chat messages the agent sees


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
