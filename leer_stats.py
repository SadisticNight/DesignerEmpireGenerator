import pickle as A
with open('estadisticas.pkl', 'rb') as B:
    C = A.load(B)
for nombre, valor in C.items():
    print(f"{nombre}: {valor}")
