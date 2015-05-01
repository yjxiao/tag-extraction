from __future__ import print_function
import numpy as np
from base import load_pickle, data_path


def main():

    print("Loading predictions ...")
    labels = load_pickle(data_path("test_lb.pkl"))
    scores = load_pickle(data_path("test_scores.pkl"))

    print("Thresholding ...")
    threshold = 0.195
    preds = scores > threshold
    
    
