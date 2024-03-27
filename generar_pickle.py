import pickle

def generar_datos():
    datos = {"celdas": []}
    for x in range(200):
        for y in range(200):
            celda = {
                "x": x,
                "y": y,
                "edificio": "",
                "atributos": {
                    "energia": 0,
                    "agua": 0,
                    "basura": 0,
                    "comida": 0,
                    "empleos": 0,
                    "residentes": 0,
                    "felicidad": 0,
                    "ambiente": 0
                }
            }
            datos["celdas"].append(celda)
    return datos

datos_generados = generar_datos()

with open('celdas.pkl', 'wb') as f:
    pickle.dump(datos_generados, f)

print("Pickle generado y guardado correctamente.")
