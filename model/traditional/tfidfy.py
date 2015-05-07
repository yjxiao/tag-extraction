from __future__ import print_function
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from base import *

def main():

    # data directory
    data_dir = "data/"
    model_dir = "model/"

    # read in training data
    train_data = []
    train_labels = []
    with open(data_path("train.tsv", data_dir), "r") as f:
        for line in f:
            content, labels = line.strip().split("\t")
            train_data.append(content)
            train_labels.append(labels.split(" "))

    # fit tfidf vectorizer on training data
    print("Fitting tfidf on training data ...")
    tv = TfidfVectorizer(stop_words="english", min_df=3)
    train_feats = tv.fit_transform(train_data)
    
    # read in test data
    
    test_data = []
    test_labels = []
    with open(data_path("test.tsv", data_dir), "r") as f:
        for line in f:
            content, labels = line.strip().split("\t")
            test_data.append(content)
            test_labels.append(labels.split(" "))

    # transform test data to tfidf vectors
    print("Transforming test data ...")
    test_feats = tv.transform(test_data)

    print("Saving ...")
    save_pickle(data_path("train.pkl", data_dir), train_feats)
    save_pickle(data_path("test.pkl", data_dir), test_feats)
    
    print("Binarizing labels ...")
    mlb = MultiLabelBinarizer()
    train_lb = mlb.fit_transform(train_labels)
    test_lb = mlb.transform(test_labels)
    
    print("Saving ...")
    save_pickle(data_path("train_lb.pkl", data_dir), train_lb)
    save_pickle(data_path("test_lb.pkl", data_dir), test_lb)
    
if __name__ == "__main__":
    main()
