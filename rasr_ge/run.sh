#!/bin/bash
set -e
python3 -m venv .venv
./.venv/bin/pip install torch pyyaml yfinance pandas numpy PyWavelets tqdm scipy scikit-learn
./.venv/bin/python data_pipeline.py > pipeline.log 2>&1
