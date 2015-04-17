import csv, cPickle
from collections import Counter

def count(labelsets):
    """ count number of appearances of each label in the labelsets """
    
    cntr = Counter()
    for labelset in labelsets:
        cntr.update(labelset)

    return cntr

def label_filter(labelsets, threshold=20):
    """ filter out labelsets that contain labels appear less than threshold times 
    Args:
        labelsets: list of sets containing labels assigned 
        threshold: minimum occurances required to remain in our perfectly selected training dataset
    
    Return:
        indeces: indeces of the posts selected
    """
    cntr = count(labelsets)    # counter storing label counts
    qualified = set()          # set storing qualified labels

    for label in cntr:
        if cntr[label] >= threshold:
            qualified.add(label)
    
    indeces = []
    q_labels = []
    for i in xrange(len(labelsets)):
        for label in labelsets[i]:
            if label not in qualified:
                break
        else:
            indeces.append(i)
            q_labels.append(labelsets[i+1])
            
    return indeces, q_labels, len(qualified)
            

def main():

    labelsets = []

    with open("Train_m.csv", 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for line in reader:
            labelsets.append(line[3].strip().split())

    threshold = 1000
    idx, labels, n_lab = label_filter(labelsets, threshold)
    
    print "number of qualified labels: {}".format(n_lab)
    print "number of qualified posts: {}".format(len(idx))

    with open("labelsets_{}.pkl".format(threshold), "wb") as f:
        cPickle.dump(labels, f)

    with open("idx_{}.txt".format(threshold), "w") as f:
        f.write("\n".join(map(str, idx)))


if __name__ == "__main__":
    main()
    
