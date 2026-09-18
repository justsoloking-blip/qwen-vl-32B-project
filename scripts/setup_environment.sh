#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Linux ]] || { echo 'The inference runtime requires Linux with NVIDIA CUDA GPUs.' >&2; exit 1; }
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
# This is a complete pip-freeze snapshot, not a set of top-level constraints.
# --no-deps intentionally preserves the explicitly validated cuDNN override.
.venv/bin/python -m pip install --no-deps -r environment/requirements.lock.txt
.venv/bin/python -c 'import torch,sglang; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available())'
