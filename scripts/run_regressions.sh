#!/usr/bin/env bash
set -euo pipefail
MODEL_RUNTIME="${MODEL_RUNTIME:-runtime/dp-attention-fixed}"
export MODEL_RUNTIME
export PYTHONPATH="$MODEL_RUNTIME/overlay${PYTHONPATH:+:$PYTHONPATH}"
"$MODEL_RUNTIME/venv/bin/python" tests/test_dp_mrope_padding.py
"$MODEL_RUNTIME/venv/bin/python" tests/test_dp_vision_dispatch.py
