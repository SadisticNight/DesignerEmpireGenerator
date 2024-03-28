import pickle as A
with open('edificios.pkl','rb')as B:C=A.load(B)
for D in C.values():print(D)