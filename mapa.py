import matplotlib.pyplot as plt
import numpy as np
import pickle

# Cargar los datos
with open('celdas.pkl', 'rb') as file:
    celdas_data = pickle.load(file)

with open('edificios.pkl', 'rb') as file:
    edificios_data = pickle.load(file)

# Crear una matriz llena de unos para la imagen (esto dará un fondo blanco)
image = np.ones((200, 200, 3))

# Llenar la imagen con los colores correspondientes
for celda in celdas_data['celdas']:
    x = celda['x']
    y = celda['y']
    edificio = celda['edificio']
    hash_value = celda['hash']
    if edificio in edificios_data and hash_value:
        # Normalizar el color al rango [0, 1]
        color = [value / 255 for value in edificios_data[edificio]['color']]
        image[x, y] = color

# Mostrar la imagen
plt.imshow(image)
plt.show()
