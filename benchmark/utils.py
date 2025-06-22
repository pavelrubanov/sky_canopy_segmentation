import numpy as np

def calculate_real_percentage(mask):
    total_pixels = mask.size
    all_pixels_sum = np.sum(mask)
    percent = (all_pixels_sum / (total_pixels * 255)) * 100
    return percent