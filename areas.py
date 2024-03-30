import pickle

_D='celdas'
_C='celdas.pkl'
_E='felicidad'
_S='ambiente'
_V='atributos'

class Area:

    area_afectada = set()

    @staticmethod
    def area_decoracion(posicion_edificio, radio=4, NUM_CELDAS=200):
        x_centro, y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        Area.area_afectada.clear()

        for x in range(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1)):
            for y in range(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1)):
                if (x - x_centro)**2 + (y - y_centro)**2 <= radio**2 and (x, y) != posicion_edificio:
                    Area.area_afectada.add((x, y))
                    
        print(Area.area_afectada)
        return list(Area.area_afectada)


    @staticmethod
    def actualizar_celdas(edificio_seleccionado, edificios):
        area_afectada_dict = {coord: None for coord in Area.area_afectada}
        if edificio_seleccionado == 'decoracion':
            with open(_C, 'rb') as file:
                celdas_data = pickle.load(file)
            for celda_coords in area_afectada_dict:
                celda = next((c for c in celdas_data[_D] if (c['x'], c['y']) == celda_coords), None)
                if celda is not None:
                    atributos_edificio = edificios[edificio_seleccionado].to_dict()
                    celda[_V][_E] += atributos_edificio[_E]
                    celda[_V][_S] += atributos_edificio[_S]
            with open(_C, 'wb') as file:
                pickle.dump(celdas_data, file)
            print(f"Área rellenada")


