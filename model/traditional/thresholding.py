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
    labels = sparse.csr_matrix(load_pickle(data_path("train_lb.pkl", data_dir)))
    scores = load_pickle(data_path("train_scores.pkl", data_dir))

    print("Testing thresholds")
    thresholds = np.arange(0, 1, 0.01)
    f = open(data_path("thresholding_results.txt", data_dir), "w")
    f.write("threshold\tf_macro\tf_micro\tprecision\trecall\tmean class accuracy\texact match\n")
    for threshold in thresholds:
        print("Calculating threshold {0}".format(threshold))
        res = evaluate(labels, scores, threshold)
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(threshold, res["f_macro"], res["f_micro"],\
                                                       res["precision"], res["recall"],\
                                                       1-res["class_accuracy"], res["exact_match"]))

    f.close()

if __name__ == "__main__":
    main()
    
