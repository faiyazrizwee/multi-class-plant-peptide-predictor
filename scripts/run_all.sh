#!/bin/bash
set -e

echo "Starting Data Prep..."
./venv/bin/python scripts/01_data_prep.py

echo "Starting Feature Extraction..."
./venv/bin/python scripts/02_extract_feat.py

echo "Starting Classical ML Baseline..."
./venv/bin/python scripts/03_train_ml.py

echo "Starting PLM Fine-Tuning..."
./venv/bin/python scripts/04_train_plm.py

echo "Starting Final Evaluation..."
./venv/bin/python scripts/05_evaluate.py

echo "Pipeline Finished!"
