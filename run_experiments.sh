#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Aktywacja venv jeśli jest
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi


python scripts/preprocess_data.py
python scripts/train.py 
