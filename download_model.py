"""
download_model.py — Downloads the GGUF model from HuggingFace if not present.
Runs as a Docker service before the LLM server starts.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    model_file = os.environ.get("LLM_MODEL_FILE")
    model_repo = os.environ.get("LLM_MODEL_REPO")

    if not model_file:
        print("ERROR: LLM_MODEL_FILE not set in .env")
        sys.exit(1)

    model_path = Path("/models") / model_file

    if model_path.exists():
        print(f"Model already exists: {model_path} — skipping download.")
        return

    if not model_repo:
        print(f"ERROR: Model not found at {model_path}")
        print("  Set LLM_MODEL_REPO in .env to enable auto-download.")
        print("  Or manually place your .gguf file in the models/ folder.")
        sys.exit(1)

    print(f"Downloading {model_file} from {model_repo} ...")
    print("This may take a few minutes depending on your connection.\n")

    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir="/models",
    )
    print(f"\nDownload complete: {model_path}")


if __name__ == "__main__":
    main()
