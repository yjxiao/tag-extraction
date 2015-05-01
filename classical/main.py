from __future__ import print_function
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import sys, getopt
from base import *

def main(argv):
    
    try:
        opts, args = getopt.getopt(argv, "", ["size=", "epochs="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))
        sys.exit()

    # default values
    data_dir = "data/34"
    model_dir = "model/34"
    n_epochs = 5

    for o, a in opts:
        if o == "--size":
            if a == 'large':
                data_dir = "data/2150"
                model_dir = "model/2150"
                print("Training with large data (2150 classes)")
        elif o == "--epochs":
            try:
                n_epochs = int(a)
            except:
                print("Invalid specification of epochs!")
                sys.exit()
            if n_epochs <= 0:
                print("Invalid specification of epochs!")
                sys.exit()

    print("Loading training data ...")
    train_data = load_pickle(data_path("train.pkl", data_dir))
    train_labels = load_pickle(data_path("train_lb.pkl", data_dir))

    print("Preparing model ...")
    clf = OneVsRestClassifier(SGDClassifier(loss="modified_huber", verbose=True, n_iter=n_epochs))
    
    print("Training ...")
    clf.fit(train_data, train_labels)

    print("Saving model ...")
    save_pickle(data_path("ovr_clf.mod", model_dir), clf)

if __name__ == "__main__":
    main(sys.argv[1:])
