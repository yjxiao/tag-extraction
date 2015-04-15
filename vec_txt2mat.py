import numpy as np
import sys
from scipy.io import savemat


def read_word_vec(in_path):
    vecs = {}
    with open(in_path, 'rb') as f:
        line = f.readline().strip().split(' ')
        vecs[line[0]] = np.array(line[1:], dtype='float32')
    return vecs


def main(args):
    vecs = read_word_vec(str(args[0]))
    savemat(str(args[1]), vecs)


if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    main(sys.argv[1:])
