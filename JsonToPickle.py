import json
import pickle

# Ruta del archivo JSON
json_file_path = 'edificios.json'

# Cargar los datos desde el archivo JSON
with open(json_file_path, 'r') as json_file:
    edificios_data = json.load(json_file)

# Ahora, convertimos estos datos a un archivo pickle
pickle_file_path = 'edificios.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(edificios_data, pickle_file)

pickle_file_path