import cPickle
import os

__all__ = ["load_pickle", "save_pickle", "data_path"]

def load_pickle(filename):
    with open(filename) as f:
        res = cPickle.load(f)
    return res

def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        cPickle.dump(obj, f)

def data_path(filename, data_dir="data"):
    return os.path.join(data_dir, filename)
