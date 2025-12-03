# areas.py

"""
Clase Area: Calcula áreas de influencia y cobertura para edificios,
actualizando dinámicamente felicidad, ambiente y servicios en el mensaje builder de celdas.
"""

import math
import numpy as np
import itertools
from bd_celdas import write_capnp_file

# Radios predefinidos para área afectada (edificios simples)
RADIOS_AREA = {
    'residencia': 4,
    'taller_togas': 4,
    'herreria': 4,
    'lecheria': 4,
    'refineria': 4,
    'policia': 4,
    'bombero': 4,
    'colegio': 4,
    'hospital': 4,
    'decoracion': 4,
}

# Radios para edificios 2x2
RADIOS_AREA_2X2 = {
    'depuradora': 4,
    'agua': 4,
}

# Radios para zonas de cobertura de servicios
RADIOS_SERVICIOS = {
    'policia': 13,
    'bombero': 13,
    'colegio': 13,
    'hospital': 11,
}

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
    def calcular_area(x_centro, y_centro, radio, NUM_CELDAS):
        x_range = np.arange(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1))
        y_range = np.arange(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1))
        xg, yg = np.meshgrid(x_range, y_range, indexing='ij')
        mask = (xg - x_centro)**2 + (yg - y_centro)**2 <= radio**2
        return set(zip(xg[mask].ravel(), yg[mask].ravel()))

    @staticmethod
    def area_afectada_simple(edificio, posicion, NUM_CELDAS=200):
        radio = RADIOS_AREA.get(edificio.lower(), 0)
        a = Area()
        a.x_centro, a.y_centro = posicion
        a.cords_edificio.add(tuple(posicion))
        a.max_radio_afectado = radio
        a.area_afectada = Area.calcular_area(a.x_centro, a.y_centro, radio, NUM_CELDAS)
        return list(a.area_afectada)

    @staticmethod
    def area_afectada_2x2(edificio, posicion, NUM_CELDAS=200):
        radio = RADIOS_AREA_2X2.get(edificio.lower(), 0)
        a = Area()
        x, y = posicion
        a.x_centro, a.y_centro = x + 0.5, y + 0.5
        a.cords_edificio.add(tuple(posicion))
        a.max_radio_afectado = radio
        prod = itertools.product(
            range(max(0, int(a.x_centro - radio - 0.5)), min(NUM_CELDAS, int(a.x_centro + radio + 1.5))),
            range(max(0, int(a.y_centro - radio - 0.5)), min(NUM_CELDAS, int(a.y_centro + radio + 1.5)))
        )
        a.area_afectada = {
            (xi, yi) for xi, yi in prod
            if (xi - a.x_centro)**2 + (yi - a.y_centro)**2 <= radio**2
        }
        return list(a.area_afectada)

    @staticmethod
    def zona_cubierta_por_edificio(edificio, posicion, NUM_CELDAS=200):
        radio = RADIOS_SERVICIOS.get(edificio.lower(), 0)
        a = Area()
        a.x_centro, a.y_centro = posicion
        a.cords_edificio.add(tuple(posicion))
        a.max_radio_cubierto = radio
        a.area_cubierta = Area.calcular_area(a.x_centro, a.y_centro, radio, NUM_CELDAS)
        return list(a.area_cubierta)

    @staticmethod
    def actualizar_celdas(edificio, datos, mensaje_builder, centro, area):
        # Actualiza felicidad y ambiente en todo el área
        R_max = max(math.sqrt((x - centro[0])**2 + (y - centro[1])**2) for x, y in area)
        for x, y in area:
            distancia = math.sqrt((x - centro[0])**2 + (y - centro[1])**2)
            factor = 0.5 + 0.5 * (1 - min(distancia / R_max, 1))
            cel = mensaje_builder.get_celda(x, y)
            cel.felicidad += round(datos['felicidad'] * factor)
            cel.ambiente  += round(datos['ambiente']  * factor)
        return mensaje_builder

    @staticmethod
    def actualizar_servicios(edificio, mensaje_builder, origen):
        # Actualiza servicios según tipo de edificio y cobertura
        coords = Area.zona_cubierta_por_edificio(edificio, origen)
        for x, y in coords:
            cel = mensaje_builder.get_celda(x, y)
            tipo = edificio.lower()
            if tipo == 'policia':
                cel.servicios.seguridad += 1
            elif tipo == 'bombero':
                cel.servicios.incendio += 1
            elif tipo == 'colegio':
                cel.servicios.educacion += 1
            elif tipo == 'hospital':
                cel.servicios.salud += 1
        return mensaje_builder

def actualizar_area_y_guardar(
    edificio, datos, mensaje_builder, centro, area, board,
    es_2x2=False, origen=None, filename="celdas.bin"
):
    # Asegura casillas en el board
    for coord in area:
        if coord not in board:
            board[coord] = ("", None)

    # Actualiza atributos de felicidad y ambiente
    if es_2x2:
        mensaje_builder = Area.actualizar_celdas_2x2(
            edificio, datos, mensaje_builder, origen, area
        )
    else:
        mensaje_builder = Area.actualizar_celdas(
            edificio, datos, mensaje_builder, centro, area
        )

    # Actualiza servicios de cobertura
    origen_coord = origen if es_2x2 else centro
    mensaje_builder = Area.actualizar_servicios(edificio, mensaje_builder, origen_coord)

    # Persiste cambios en celdas.bin
    write_capnp_file(filename, mensaje_builder)
    return mensaje_builder
