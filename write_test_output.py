__author__ = 'ramkishore'

with open('output_prediction.txt') as f:
    lines = f.read().splitlines()

index = 0
predictions = open("data_predictions_Helpful.txt", 'w')
for l in open("data_pairs_Helpful.txt"):
  if l.startswith("userID"):
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf * float(lines[index])) + '\n')
  index += 1
predictions.close()
