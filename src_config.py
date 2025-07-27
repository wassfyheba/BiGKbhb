# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 14:17:54 2025

@author: heba_
"""


import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'window_size': 41,
    'encoding_dim': 20,
    'gru_units': 192,
    'dropout_rate': 0.4,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 60,
    'patience': 10,
    'random_state': 42
}

# Cross-validation settings
CV_CONFIG = {
    'n_splits': 10,
    'shuffle': True,
    'random_state': 42
}

# Data splitting
DATA_CONFIG = {
    'test_size': 0.1,
    'validation_split': 0.1,
    'stratify': True,
    'random_state': 42
}
EOF