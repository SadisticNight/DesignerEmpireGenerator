import pickle

# Abre el archivo 'celdas.pkl' en modo de lectura binaria
with open('celdas.pkl', 'rb') as file:
    celdas_data = pickle.load(file)

# Imprime una porción de los datos para verificar, por ejemplo, las primeras 10 celdas
for celda in celdas_data['celdas'][:100]:
    print(celda)
