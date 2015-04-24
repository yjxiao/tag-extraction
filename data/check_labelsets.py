from collections import Counter
import cPickle

with open('labelsets_50000.pkl', 'rb') as f:
    t = cPickle.load(f)
cntr = Counter()
print len(t)
for i in t:
    cntr.update(i)

print len(cntr)
