import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

def load_models(cfg, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []
    
    model_files = find_model_files(cfg)
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models