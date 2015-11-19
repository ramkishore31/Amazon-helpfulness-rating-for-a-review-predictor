__author__ = 'ramkishore'

import gzip
import json

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

data = []
with open('processed_data.json', 'w') as outfile:
    for l in readGz("train.json.gz"):
        data.append(l)
        if(l['helpful']['outOf'] > 10):
            json.dump(l, outfile)
            outfile.write('\n')


