# Run `$ python HF_TOKE=token download.py` to download the model
from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('/Data/Mistral_models/', 'Mamba-Codestral-7B-v0.1')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mamba-Codestral-7B-v0.1", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

