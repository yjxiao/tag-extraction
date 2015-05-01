from __future__ import print_function
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from scipy.sparse import csr_matrix, vstack
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

    for o, a in opts:
        if o == "--size":
            if a == 'large':
                data_dir = "data/2150"
                model_dir = "model/2150"

    print("Loading test data ...")
    test_data = load_pickle(data_path("test.pkl", data_dir))
    test_labels = load_pickle(data_path("test_lb.pkl", data_dir))

    print("Loading model ...")
    clf = load_pickle(data_path("ovr_clf.mod", model_dir))
    
    print("Testing ...")
    n_classes = test_labels.shape[1]
    n_samples = test_labels.shape[0]
    batch_size = 10000
    n_batch = n_samples // batch_size + 1
    probs = csr_matrix(np.zeros((0, n_classes)))

    for i in xrange(n_batch):
        print("==== Predicting batch {0} ====".format(i+1))
        start = i * batch_size
        end = min((i+1)*batch_size, n_samples)
        probs = vstack([probs, csr_matrix(clf.predict_proba(test_data[start:end]))])

    print("Done!")
    print("Saving predictions ...")
    save_pickle(data_path("test_scores.pkl", data_dir), probs)

if __name__ == "__main__":
    main(sys.argv[1:])
