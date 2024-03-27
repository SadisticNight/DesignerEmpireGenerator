import pickle

edificios_pkl_path = 'edificios.pkl'
with open(edificios_pkl_path, 'rb') as file:
    edificios_data = pickle.load(file)
for edificio, atributos in list(edificios_data.items()):
    print(f"Edificio: {edificio}, Atributos: {atributos}")
