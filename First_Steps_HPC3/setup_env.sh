#!/bin/bash
# One-time setup on the LOGIN node: create a Python venv in your home directory
# and download MNIST. Usage:  bash setup_env.sh
set -e
python3 -m venv ~/venvs/genai
source ~/venvs/genai/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt     # PyTorch with CUDA support is several GB (counts against your quota)
python download_data.py
