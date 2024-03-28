import json,pickle as B
C='edificios.json'
with open(C,'r')as D:E=json.load(D)
A='edificios.pkl'
with open(A,'wb')as F:B.dump(E,F)
A