import matplotlib.pyplot as plt
import numpy as np
import capnp

# Cargar los esquemas de Cap'n Proto para celdas y edificios
cc = capnp.load("celdas.capnp")
ec = capnp.load("edificios.capnp")

# Leer el mensaje Mapa desde el binario 'celdas.bin'
with open("celdas.bin", "rb") as f:
    mapa = cc.Mapa.read(f)

# Leer el mensaje Edificios desde el binario 'edificios.bin'
with open("edificios.bin", "rb") as f:
    edificios_data = ec.Edificios.read(f)

# Crear un diccionario que mapea el nombre del edificio a su objeto (del esquema)
eds = {b.nombre: b for b in edificios_data.lista}

# Crear una imagen 200x200 con fondo blanco (valores 1 en cada canal: [1,1,1])
img = np.ones((200, 200, 3))

# Iterar sobre cada celda registrada en el mapa
for cel in mapa.celdas:
    # Si la celda tiene asignado un edificio y su hash es verdadero (es decir, la celda tiene datos)
    if cel.edificio in eds and cel.hash:
        # Convertir el color del edificio a valores normalizados (0-1)
        # Se asume que cel.edificio es el nombre del edificio y que eds[cel.edificio].color es una lista de 3 números (R,G,B)
        color_normalized = [v / 255 for v in eds[cel.edificio].color]
        # En NumPy la primera dimensión es Y y la segunda es X, por lo que asignamos en (cel.y, cel.x)
        img[int(cel.y), int(cel.x)] = color_normalized

# Mostrar la imagen resultante
plt.imshow(img)
plt.title("Mapa de celdas con edificios")
plt.axis('off')
plt.show()
