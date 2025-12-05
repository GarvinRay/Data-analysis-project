#!/usr/bin/env python3
"""
Centralized file/path configuration for the Meta-AC project.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Processed dataset paths
STATS_PATH = PROCESSED_DIR / "meta_ac_stats_sampled.csv"
JSON_PATH = PROCESSED_DIR / "meta_ac_dataset_sampled.json"

# Output artifacts
PREDICTIONS_PATH = OUTPUT_DIR / "final_predictions.csv"
PLOT_PATH = OUTPUT_DIR / "meta_ac_impact.png"
MODEL_PATH = OUTPUT_DIR / "meta_ac_model.pkl"
