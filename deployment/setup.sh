#!/bin/bash

set -euxo pipefail

echo "Downloading the Whisper models to the cache directory"

# TODO: Find a way to re-use the same parameters as specified in service.py
# BentoML seems to have issues importing variables from another module in setup scripts
python3 - <<EOF
from faster_whisper.utils import download_model
download_model(
    size_or_id="small",
    cache_dir="/home/bentoml/.cache/huggingface/hub",
)
EOF