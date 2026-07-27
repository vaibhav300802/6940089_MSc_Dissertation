#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/dashboard"
cd ..
python validate_project.py
python dashboard/startup_validation.py
cd dashboard
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
