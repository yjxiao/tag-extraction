from collections import Counter
import cPickle

with open('labelsets_50000.pkl', 'rb') as f:
    t = cPickle.load(f)
cntr = Counter()
print "Number of samples left: {}".format(len(t))
for i in t:
    cntr.update(i)

print 'Number of labels left: {}'.format(len(cntr))
print 'Saving labels to file'
with open('labels.txt', 'wb') as f:
    f.write('\n'.join(cntr.keys()))
