import pickle

# Abre el archivo 'edificios.pkl' en modo de lectura binaria
with open('edificios.pkl', 'rb') as file:
    edificios_data = pickle.load(file)

# Imprime todos los datos de los edificios
for edificio in edificios_data.values():
    print(edificio)
