"""Run in the Linux project environment containing huggingface-hub==0.36.2."""
import argparse
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);a=p.parse_args()
snapshot_download(repo_id='Qwen/Qwen3-VL-32B-Instruct',revision='0cfaf48183f594c314753d30a4c4974bc75f3ccb',local_dir=a.model_dir)
