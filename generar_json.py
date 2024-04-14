import json

def generar_json():
    datos = {"celdas": []}
    for x in range(200):
        for y in range(200):
            celda = {
                "x": x,
                "y": y,
                "hash":"",
                "edificio": "",
                "tipo":"",
                "atributos": {
                    "energia": 0,
                    "agua": 0,
                    "basura": 0,
                    "comida": 0,
                    "empleos": 0,
                    "residentes": 0,
                    "felicidad": 0,
                    "ambiente": 0
                },
                "servicios": {
                    "seguridad": 0,
                    "incendio": 0,
                    "salud": 0,
                    "educacion": 0
                }
            }
            datos["celdas"].append(celda)
    return datos

json_generado = generar_json()

with open('celdas.json', 'w') as f:
    json.dump(json_generado, f, indent=2)

print("JSON generado y guardado correctamente.")