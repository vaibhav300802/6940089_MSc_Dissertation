#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -m pip install -r requirements.txt
cd ..
python dashboard/startup_validation.py
cd dashboard
streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
