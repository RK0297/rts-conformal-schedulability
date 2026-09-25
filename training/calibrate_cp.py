"""Calibration Script for Split Conformal Prediction."""
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

from models.conformal_wrapper import ConformalPredictionWrapper

def calibrate(data_npz: str, checkpoint: str, alphas: List[float] = [0.01, 0.05, 0.10, 0.20]):
    data = np.load(data_npz)
    print(f"Calibrating conformal prediction sets on {data_npz} across alphas: {alphas}...")
