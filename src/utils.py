# a file with useful functions

import numpy as np
import pandas as pd

def moving_average_single(data, period, start_idx):
    if start_idx + period > len(data):
        raise ValueError("Not enough data points left to satisfy the period.")
        
    return np.sum(data[start_idx:start_idx+period]) / period

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'valid') / period