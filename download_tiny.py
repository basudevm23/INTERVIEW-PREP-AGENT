# download_tiny.py
from huggingface_hub import snapshot_download
if __name__ == "__main__":
    snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir="./tiny_llama")
    print("Downloaded tiny_llama to ./tiny_llama")
