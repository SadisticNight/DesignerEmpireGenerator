import pickle

class Area:

    @staticmethod
    def area_decoracion(posicion_edificio, radio=4, NUM_CELDAS=200):
        x_centro, y_centro = posicion_edificio
        area = []

        for x in range(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1)):
            for y in range(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1)):
                if (x - x_centro)**2 + (y - y_centro)**2 <= radio**2:
                    area.append((x, y))
                    print(f"Celda en el área de decoración: ({x}, {y})")

        return area



    @staticmethod
    def actualizar_celdas(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios, celdas_data):
        if edificio_seleccionado == 'decoracion':
            area = Area.area_decoracion(posicion)
            for c in celdas_data:
                if isinstance(c, dict) and (c['x'], c['y']) in area:
                    print(f"Actualizando celda en ({c['x']}, {c['y']})")
                    c['atributos']['felicidad'] += edificios[edificio_seleccionado]['felicidad']
                    c['atributos']['ambiente'] += edificios[edificio_seleccionado]['ambiente']
            with open('celdas.pkl', 'wb') as file:
                pickle.dump(celdas_data, file)










