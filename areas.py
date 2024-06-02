import itertools
import pickle
from collections import defaultdict
from functools import lru_cache

_A = None
_Q = 'celdas'
_C = 'celdas.pkl'
_E = 'felicidad'
_S = 'ambiente'
_V = 'atributos'

@lru_cache(maxsize=_A)
def read_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def write_pickle_file(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

class Area:
    __slots__ = (
        'area_afectada', 'area_cubierta', 'cords_edificio', 
        'max_radio_afectado', 'max_radio_cubierto', 'x_centro', 
        'y_centro', 'max_efecto'
    )

    def __init__(self):
        self.area_afectada = set()
        self.area_cubierta = set()
        self.cords_edificio = set()
        self.max_radio_afectado = 0
        self.max_radio_cubierto = 0
        self.x_centro = 0
        self.y_centro = 0
        self.max_efecto = 100

    @staticmethod
    def area_defecto(edificio_seleccionado, posicion_edificio):
        match edificio_seleccionado:
            case 'residencia' | 'taller_togas' | 'herreria' | 'decoracion' | 'lecheria' | 'refineria' | 'policia' | 'bombero' | 'colegio' | 'hospital':
                pass
            case _:
                posicion_edificio = tuple(posicion_edificio)
                area = Area()
                area.area_afectada.add(posicion_edificio)
                area.area_cubierta.add(posicion_edificio)
                area.cords_edificio.add(posicion_edificio)
                area.x_centro, area.y_centro = posicion_edificio

    @staticmethod
    def area_afectada_(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        radios = {
            'residencia': 4, 'taller_togas': 4, 'herreria': 4,
            'lecheria': 4, 'refineria': 4, 'policia': 4,
            'bombero': 4, 'colegio': 4, 'hospital': 4
        }

        radio = radios.get(edificio_seleccionado)
        if radio is _A:
            return

        area = Area()
        area.x_centro, area.y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        area.cords_edificio.add(posicion_edificio)
        area.max_radio_afectado = radio

        area.area_afectada = {
            (x, y)
            for x, y in itertools.product(
                range(max(0, area.x_centro - radio), min(NUM_CELDAS, area.x_centro + radio + 1)),
                range(max(0, area.y_centro - radio), min(NUM_CELDAS, area.y_centro + radio + 1))
            )
            if (x - area.x_centro) ** 2 + (y - area.y_centro) ** 2 <= radio ** 2
        }
        return list(area.area_afectada)

    @staticmethod
    def area_afectada_por_edificio_2x2(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        radios = {
            'depuradora': 4,
            'agua': 4
        }

        radio = radios.get(edificio_seleccionado)
        if radio is _A:
            return

        area = Area()
        area.x_centro = posicion_edificio[0] + 0.5
        area.y_centro = posicion_edificio[1] + 0.5

        area.area_afectada = {
            (x, y)
            for x, y in itertools.product(
                range(max(0, int(area.x_centro - radio - 0.5)), min(NUM_CELDAS, int(area.x_centro + radio + 1.5))),
                range(max(0, int(area.y_centro - radio - 0.5)), min(NUM_CELDAS, int(area.y_centro + radio + 1.5)))
            )
        }

        return list(area.area_afectada)

    @staticmethod
    def area_afectada_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        radios = {
            'decoracion': 4
        }

        radio = radios.get(edificio_seleccionado)
        if radio is _A:
            return

        area = Area()
        area.x_centro, area.y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        area.cords_edificio.add(posicion_edificio)
        area.max_radio_afectado = radio

        area.area_afectada = {
            (x, y)
            for x, y in itertools.product(
                range(max(0, area.x_centro - radio), min(NUM_CELDAS, area.x_centro + radio + 1)),
                range(max(0, area.y_centro - radio), min(NUM_CELDAS, area.y_centro + radio + 1))
            )
            if (x - area.x_centro) ** 2 + (y - area.y_centro) ** 2 <= radio ** 2
        }
        return list(area.area_afectada)

    @staticmethod
    def zona_cubierta_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        radios = {
            'policia': 13,
            'bombero': 13,
            'colegio': 13,
            'hospital': 11
        }

        radio = radios.get(edificio_seleccionado)
        if radio is _A:
            return

        area = Area()
        area.x_centro, area.y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        area.cords_edificio.add(posicion_edificio)
        area.max_radio_cubierto = radio

        area.area_cubierta = {
            (x, y)
            for x, y in itertools.product(
                range(max(0, area.x_centro - radio), min(NUM_CELDAS, area.x_centro + radio + 1)),
                range(max(0, area.y_centro - radio), min(NUM_CELDAS, area.y_centro + radio + 1))
            )
            if (x - area.x_centro) ** 2 + (y - area.y_centro) ** 2 <= radio ** 2
        }
        return list(area.area_cubierta)

    @staticmethod
    def actualizar_celdas(edificio_seleccionado, edificios):
        match edificio_seleccionado:
            case 'residencia' | 'taller_togas' | 'herreria' | 'decoracion' | 'lecheria' | 'refineria' | 'policia' | 'bombero' | 'colegio' | 'hospital':
                area = Area()
                area_afectada_dict = {coord: _A for coord in area.area_afectada.copy()}  # Truco set().copy() aplicado
                
                celdas_data = read_pickle_file(_C)
                
                def actualizar_celda(celda, atributos_edificio, factor_influencia):
                    celda[_V][_E] += round(factor_influencia * atributos_edificio[_E])
                    celda[_V][_S] += round(factor_influencia * atributos_edificio[_S])

                for celda_coords in area_afectada_dict:
                    celda = next((c for c in celdas_data[_Q] if (c['x'], 'y') == celda_coords), _A)
                    if celda is not _A:
                        atributos_edificio = edificios[edificio_seleccionado].to_dict()
                        distancia_influencia = ((celda['x'] - area.x_centro)**2 + (celda['y'] - area.y_centro)**2)**0.5
                        factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / area.max_radio_afectado, 1))
                        actualizar_celda(celda, atributos_edificio, factor_influencia)
                
                write_pickle_file(_C, celdas_data)

    @staticmethod
    def actualizar_celdas_2x2(edificio_seleccionado, posicion_edificio, edificios, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'agua' | 'depuradora':
                area = Area()
                area_afectada = area.area_afectada_por_edificio_2x2(edificio_seleccionado, posicion_edificio, NUM_CELDAS)
                area_afectada_dict = {coord: _A for coord in area.area_afectada.copy()}  # Truco set().copy() aplicado

                celdas_data = read_pickle_file(_C)

                atributos_edificio = edificios[edificio_seleccionado].to_dict()

                def actualizar_celda_2x2(celda, x_centro, y_centro, atributos_edificio):
                    distancia_influencia = ((celda['x'] - x_centro)**2 + (celda['y'] - y_centro)**2)**0.5
                    factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / (area.max_radio_afectado + 1), 1))
                    celda[_V][_E] += round(factor_influencia * atributos_edificio[_E])
                    celda[_V][_S] += round(factor_influencia * atributos_edificio[_S])

                x_centro = posicion_edificio[0] + 0.5
                y_centro = posicion_edificio[1] + 0.5

                for celda_coords in area_afectada_dict:
                    celda = next((c for c in celdas_data[_Q] if c['x'] == celda_coords[0] and c['y'] == celda_coords[1]), _A)
                    if celda is not _A:
                        actualizar_celda_2x2(celda, x_centro, y_centro, atributos_edificio)

                write_pickle_file(_C, celdas_data)

    @staticmethod
    def servicios_cubiertos(edificio_seleccionado, edificios):
        area = Area()
        area_cubierta_dict = {coord: _A for coord in area.area_cubierta.copy()}  # Truco set().copy() aplicado
        celdas_data = read_pickle_file(_C)

        servicios = {
            'policia': 'seguridad',
            'bombero': 'incendio',
            'colegio': 'educacion',
            'hospital': 'salud'
        }

        servicio = servicios.get(edificio_seleccionado)
        if servicio is _A:
            return

        def actualizar_servicio(celda, servicio):
            distancia_influencia = ((celda['x'] - area.x_centro)**2 + (celda['y'] - area.y_centro)**2)**0.5
            factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / area.max_radio_cubierto, 1))
            incremento = round(factor_influencia * area.max_efecto)
            celda['servicios'][servicio] = min(area.max_efecto, max(celda['servicios'].get(servicio, 0), incremento))

        for celda_coords in area_cubierta_dict:
            celda = next((c for c in celdas_data[_Q] if (c['x'], 'y') == celda_coords), _A)
            if celda is not _A:
                actualizar_servicio(celda, servicio)

        write_pickle_file(_C, celdas_data)
