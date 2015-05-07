from __future__ import division, print_function
import numpy as np
from scipy import sparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from base import load_pickle, data_path

def evaluate(labels, scores, threshold):
    """ """
    preds = scores > threshold
    f_micro = f1_score(labels, preds, average="micro")
    f_macro = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    exact_match = accuracy_score(labels, preds)

    n_classes = scores.shape[1]
    n_samples = scores.shape[0]
    class_acc = np.zeros(n_classes)
    for i in xrange(n_classes):
        n_false = (labels[:, i] != preds[:, i]).sum()
        class_acc[i] = n_false / n_samples
        
    return dict(f_micro=f_micro, f_macro=f_macro, exact_match=exact_match, precision=precision,\
                    recall=recall, class_accuracy=class_acc.mean())

def main():

    data_dir = "data/34/"
    print("Loading predictions ...")
    labels = sparse.csr_matrix(load_pickle(data_path("test_lb.pkl", data_dir)))
    scores = load_pickle(data_path("test_scores.pkl", data_dir))

    print("Testing on test data ...")
    threshold = 0.28
    res = evaluate(labels, scores, threshold)
    
    print("f_macro\tf_micro\tprecision\trecall\tmean class acc\texact match")
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(res["f_macro"], res["f_micro"],\
                                                       res["precision"], res["recall"],\
                                                       1-res["class_accuracy"], res["exact_match"]))


if __name__ == "__main__":
    main()
    
