import pygame, sys, numpy as np
from menu import Menu
from edificios import edificios
from atributos import Atributo
from generar_pickle import generar_datos
from crear_stats import generar_datos_stats
import os, pickle, hashlib, time
from condiciones import Condiciones
from areas import Area
from estadisticas import Stats
import random
from collections import defaultdict
from functools import lru_cache
import itertools

TECLA_IZQUIERDA = pygame.K_LEFT
TECLA_DERECHA = pygame.K_RIGHT
TECLA_ARRIBA = pygame.K_UP
TECLA_ABAJO = pygame.K_DOWN

ANCHO_VENTANA, ALTO_VENTANA = 600, 600
TAMANO_CELDA = ANCHO_VENTANA // 200
NUM_CELDAS = 200
VERDE = 0, 255, 0
BLANCO = 255, 255, 255

_N = None
_F = 'No se puede colocar el edificio aquí'
_Y = 'edificio'
_Q = 'celdas'
_H = 'hash'
_C = 'celdas.pkl'
_W = 'estadisticas.pkl'
_A = 'energia'
_T = 'agua'
_R = 'basura'
_I = 'comida'
_B = 'empleos'
_U = 'residentes'
_E = 'felicidad'
_S = 'ambiente'
_V = 'atributos'
_Z = 'tipo'

if not os.path.exists(_C): generar_datos()
if not os.path.exists(_W): generar_datos_stats()

pygame.init()
ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
pygame.display.set_caption('Juego de Construcción de Ciudades')
reloj = pygame.time.Clock()
menu = Menu(ANCHO_VENTANA, ALTO_VENTANA)
posicion_usuario = [NUM_CELDAS // 2, NUM_CELDAS // 2]
ultima_posicion = list(posicion_usuario)
mapa = np.full((NUM_CELDAS, NUM_CELDAS), _N)
edificio_seleccionado = _N
stats = Stats()

@lru_cache(maxsize=None)  # Truco Uso de lru_cache aplicado
def read_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def write_pickle_file(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

@lru_cache(maxsize=None)  # Truco Uso de lru_cache aplicado
def generar_hash(edificio):
    return hashlib.sha256((edificio + str(time.time())).encode()).hexdigest()

def actualizar_seleccion():
    global tamanio_cursor, edificio_seleccionado
    match edificio_seleccionado:
        case None:
            tamanio_cursor = (1, 1)
        case _:
            tamanio_cursor = edificios[edificio_seleccionado].tamanio

def dibujar_celda(fila, columna, atributo):
    tamanio_edificio = atributo.tamanio
    if atributo.color == VERDE:  # Si el atributo es el cursor
        rect = pygame.Rect(columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA * tamanio_edificio[0], TAMANO_CELDA * tamanio_edificio[1])
    else:  # Si el atributo es un edificio
        match tamanio_edificio:
            case [2, 2]:
                rect = pygame.Rect(columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA * (tamanio_edificio[0] - 1), TAMANO_CELDA * (tamanio_edificio[1] - 1))
            case [1, 1]:
                rect = pygame.Rect(columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA * tamanio_edificio[0], TAMANO_CELDA * tamanio_edificio[1])
            case _:
                pass
    pygame.draw.rect(ventana, atributo.color, rect, 0)

def dibujar_cuadricula():
    global tamanio_cursor
    ventana.fill(BLANCO)
    actualizar_seleccion()
    
    # Dibujar todos los edificios primero
    for fila, columna in itertools.product(range(NUM_CELDAS), repeat=2):  # Truco Uso de Iteradores y Generadores aplicado
        match mapa[fila][columna]:
            case None:
                pass
            case _:
                atributo_edificio = edificios[mapa[fila][columna]]
                dibujar_celda(fila, columna, atributo_edificio)

    # Dibujar el cursor del usuario
    dibujar_celda(posicion_usuario[0], posicion_usuario[1], Atributo(VERDE, 0, 0, 0, 0, 0, 0, None, 0, 0, list(tamanio_cursor)))

def actualizar_pantalla():
    dibujar_cuadricula()
    menu.dibujar(ventana)
    pygame.display.flip()

def actualizar_area_y_celdas(edificio_seleccionado, posicion_usuario, NUM_CELDAS, edificios):
    Area.area_defecto(edificio_seleccionado, posicion_usuario)
    match edificio_seleccionado:
        case 'residencia' | 'taller_togas' | 'herreria' | 'lecheria' | 'refineria':
            Area.area_afectada_(edificio_seleccionado, posicion_usuario, NUM_CELDAS)
            Area.actualizar_celdas(edificio_seleccionado, edificios)
        case 'policia' | 'bombero' | 'colegio' | 'hospital':
            Area.area_afectada_(edificio_seleccionado, posicion_usuario, NUM_CELDAS)
            Area.actualizar_celdas(edificio_seleccionado, edificios)
            Area.zona_cubierta_por_edificio(edificio_seleccionado, posicion_usuario, NUM_CELDAS)
            Area.servicios_cubiertos(edificio_seleccionado, edificios)
        case 'agua' | 'depuradora':
            Area.area_afectada_por_edificio_2x2(edificio_seleccionado, posicion_usuario, NUM_CELDAS)
            Area.actualizar_celdas_2x2(edificio_seleccionado, posicion_usuario, edificios, NUM_CELDAS)
        case 'decoracion':
            Area.area_afectada_por_edificio(edificio_seleccionado, posicion_usuario, NUM_CELDAS)
            Area.actualizar_celdas(edificio_seleccionado, edificios)                                                
        case _:
            pass
    stats.procesar_estadisticas()

def llenar_mapa_aleatoriamente():
    tipos_edificios = list(edificios.keys())
    archivo_celdas = _C
    print(f"Archivo usado: {archivo_celdas}")
    Total1 = 0
    Total2 = 0
    Totales = 0
    Libres = 40000
    Ocupadas = 0

    while Libres > 0:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE:
                return

        coordenadas = itertools.product(range(NUM_CELDAS), repeat=2)  # Truco Uso de Iteradores y Generadores aplicado
        for fila, columna in coordenadas:
            match mapa[fila][columna]:
                case None:
                    edificios_disponibles = set(tipos_edificios).copy()
                    while edificios_disponibles:
                        edificio_seleccionado = random.choice(list(edificios_disponibles))
                        match Condiciones.condiciones(edificio_seleccionado, (fila, columna), mapa, NUM_CELDAS, edificios):
                            case True:
                                tamanio_edificio = edificios[edificio_seleccionado].tamanio
                                puede_colocar = False
                                match tamanio_edificio:
                                    case [2, 2]:
                                        if fila + 1 < NUM_CELDAS and columna + 1 < NUM_CELDAS:
                                            puede_colocar = all(
                                                mapa[fila + f][columna + c] is _N
                                                for f, c in itertools.product(range(2), repeat=2)  # Truco Uso de Iteradores y Generadores aplicado
                                            )
                                        else:
                                            print("No se puede colocar el edificio aquí, está fuera de los límites del mapa")
                                            edificios_disponibles.remove(edificio_seleccionado)
                                            continue
                                    case [1, 1] if mapa[fila][columna] is _N:
                                        puede_colocar = True
                                    case _:
                                        pass

                                match puede_colocar:
                                    case True:
                                        hash_edificio = generar_hash(edificio_seleccionado)
                                        atributos_edificio = edificios[edificio_seleccionado].to_dict()
                                        coordenadas_edificio = [(fila + f, columna + c) for f, c in itertools.product(range(tamanio_edificio[0]), range(tamanio_edificio[1]))]
                                        
                                        celdas_data = read_pickle_file(_C)

                                        for x, y in coordenadas_edificio:
                                            mapa[x][y] = edificio_seleccionado
                                            celda = next((celda for celda in celdas_data[_Q] if celda['x'] == x and celda['y'] == y), _N)
                                            match celda:
                                                case None:
                                                    pass
                                                case _:
                                                    celda[_Y] = edificio_seleccionado
                                                    celda[_H] = hash_edificio
                                                    celda[_Z] = atributos_edificio[_Z]
                                                    celda[_V] = {key: atributos_edificio[key] for key in [_A, _T, _R, _I, _B, _U, _E, _S]}
                                        
                                        write_pickle_file(_C, celdas_data)

                                        print(f"Edificio {edificio_seleccionado} colocado en las coordenadas: {coordenadas_edificio}")
                                        actualizar_area_y_celdas(edificio_seleccionado, (fila, columna), NUM_CELDAS, edificios)
                                        match tamanio_edificio:
                                            case [2, 2]:
                                                Total2 += 1
                                                Libres -= 4
                                                Ocupadas += 4
                                            case [1, 1]:
                                                Total1 += 1
                                                Libres -= 1
                                                Ocupadas += 1
                                        break
                                    case False:
                                        edificios_disponibles.remove(edificio_seleccionado)
                            case False:
                                edificios_disponibles.remove(edificio_seleccionado)
                    else:
                        print('No se pudo colocar ningún edificio en esta celda')
                case _:
                    print(_F)
            Totales = Total1 + Total2
            print(f"Total 1x1: {Total1}")
            print(f"Total 2x2: {Total2}")
            print(f"Edificios en total: {Totales}")
            print(f"Espacios ocupados: {Ocupadas}")
            print(f"Espacios libres: {Libres}")
            actualizar_pantalla()
            pygame.event.pump()

llenar_mapa_aleatoriamente()

dibujar_cuadricula()
pygame.display.flip()


while True:
    teclas_presionadas = pygame.key.get_pressed()
    ultima_posicion = list(posicion_usuario)

    if not menu.menu_activo:
        tamanio_actual = edificios[edificio_seleccionado].tamanio if edificio_seleccionado is not _N else (1, 1)

        if teclas_presionadas[TECLA_IZQUIERDA] and posicion_usuario[1] > 0:
            posicion_usuario[1] -= 1
        if teclas_presionadas[TECLA_DERECHA] and posicion_usuario[1] < NUM_CELDAS - tamanio_actual[1]:
            posicion_usuario[1] += 1
        if teclas_presionadas[TECLA_ARRIBA] and posicion_usuario[0] > 0:
            posicion_usuario[0] -= 1
        if teclas_presionadas[TECLA_ABAJO] and posicion_usuario[0] < NUM_CELDAS - tamanio_actual[0]:
            posicion_usuario[0] += 1

        if ultima_posicion != posicion_usuario:
            actualizar_pantalla()

    for evento in pygame.event.get():
        match evento.type:
            case pygame.QUIT:
                pygame.quit()
                sys.exit()

            case pygame.KEYDOWN:
                if evento.key == pygame.K_n:
                    x, y = posicion_usuario
                    match edificio_seleccionado:
                        case None:
                            print(f"Coordenada inicial 1x1: ({x}, {y})")
                        case _:
                            tamanio_edificio = edificios[edificio_seleccionado].tamanio
                            match tamanio_edificio:
                                case [2, 2]:
                                    print(f"Coordenadas 2x2: ({x}, {y}), ({x+1}, {y}), ({x}, {y+1}), ({x+1}, {y+1})")
                                case [1, 1]:
                                    print(f"Coordenada 1x1: ({x}, {y})")
                                case _:
                                    pass          

                if evento.key == pygame.K_m:
                    menu.toggle_menu()
                    actualizar_pantalla()

                if evento.key == pygame.K_RETURN and menu.menu_activo:
                    edificio_seleccionado = menu.opciones[menu.indice_seleccionado]
                    if edificio_seleccionado is not _N:
                        tamanio_edificio = edificios[edificio_seleccionado].tamanio
                        if posicion_usuario[1] + tamanio_edificio[1] > NUM_CELDAS:
                            posicion_usuario[1] = NUM_CELDAS - tamanio_edificio[1]
                        if posicion_usuario[0] + tamanio_edificio[0] > NUM_CELDAS:
                            posicion_usuario[0] = NUM_CELDAS - tamanio_edificio[0]

                    actualizar_seleccion()
                    menu.toggle_menu()
                    actualizar_pantalla()
                    print(f"{edificio_seleccionado} seleccionado")

                if evento.key in (pygame.K_KP_PLUS, pygame.K_PLUS):
                    match edificio_seleccionado:
                        case None:
                            print('Debe seleccionar un edificio')
                        case _:
                            if Condiciones.condiciones(edificio_seleccionado, posicion_usuario, mapa, NUM_CELDAS, edificios):
                                tamanio_edificio = edificios[edificio_seleccionado].tamanio
                                puede_colocar = False

                                match tamanio_edificio:
                                    case [2, 2] if posicion_usuario[0] + 1 < NUM_CELDAS and posicion_usuario[1] + 1 < NUM_CELDAS:
                                        puede_colocar = all(
                                            mapa[posicion_usuario[0] + f][posicion_usuario[1] + c] is _N
                                            for f, c in itertools.product(range(2), repeat=2)  # Truco Uso de Iteradores y Generadores aplicado
                                        )
                                    case [1, 1] if mapa[posicion_usuario[0]][posicion_usuario[1]] is _N:
                                        puede_colocar = True

                                match puede_colocar:
                                    case True:
                                        hash_edificio = generar_hash(edificio_seleccionado)
                                        atributos_edificio = edificios[edificio_seleccionado].to_dict()
                                        coordenadas_edificio = [(posicion_usuario[0] + f, posicion_usuario[1] + c) for f, c in itertools.product(range(tamanio_edificio[0]), range(tamanio_edificio[1]))]  # Truco Uso de Iteradores y Generadores aplicado

                                        # Leer el archivo pickle una vez
                                        celdas_data = read_pickle_file(_C)

                                        for x, y in coordenadas_edificio:
                                            mapa[x][y] = edificio_seleccionado
                                            celda = next((celda for celda in celdas_data[_Q] if celda['x'] == x and celda['y'] == y), _N)  # Truco Optimización de Búsqueda en Listas aplicado
                                            match celda:
                                                case None:
                                                    pass
                                                case _:
                                                    celda[_Y] = edificio_seleccionado
                                                    celda[_H] = hash_edificio
                                                    celda[_Z] = atributos_edificio[_Z].value
                                                    celda[_V] = {key: atributos_edificio[key] for key in [_A, _T, _R, _I, _B, _U, _E, _S]}
                                        
                                        # Escribir el archivo pickle una vez
                                        write_pickle_file(_C, celdas_data)

                                        print(f"Edificio {edificio_seleccionado} colocado en las coordenadas: {coordenadas_edificio}")
                                        actualizar_area_y_celdas(edificio_seleccionado, posicion_usuario, NUM_CELDAS, edificios)
                                    case _:
                                        print(_F)
                            actualizar_pantalla()

                if evento.key in (pygame.K_KP_MINUS, pygame.K_MINUS):
                    match tamanio_cursor:
                        case [1, 1]:
                            # Leer el archivo pickle una vez
                            celdas_data = read_pickle_file(_C)
                            hash_a_eliminar = _N
                            atributos_edificio = _N

                            # Encontrar el edificio a eliminar
                            for celda in celdas_data[_Q]:
                                if celda['x'] == posicion_usuario[0] and celda['y'] == posicion_usuario[1]:
                                    if celda[_Y] != '':
                                        hash_a_eliminar = celda[_H]
                                        atributos_edificio = edificios[celda[_Y]].to_dict()
                                        break
                            else:
                                print('No hay edificio en esta celda')
                                continue

                            if hash_a_eliminar:
                                # Eliminar el edificio de las celdas correspondientes
                                for celda in celdas_data[_Q]:
                                    if celda[_H] == hash_a_eliminar:
                                        celda[_Y] = ''
                                        celda[_H] = ''
                                        celda[_Z] = ''
                                        celda[_V] = {key: celda[_V][key] - atributos_edificio[key] for key in [_A, _T, _R, _I, _B, _U, _E, _S]}
                                        mapa[celda['x'], celda['y']] = _N

                                # Escribir el archivo pickle una vez
                                write_pickle_file(_C, celdas_data)
                                print('Edificio eliminado')

                        case [2, 2]:
                            print('El cursor debe ser 1x1')
                        case _:
                            print('El cursor debe ser 1x1')

                    actualizar_pantalla()

                if menu.menu_activo:
                    menu.manejar_evento(evento)

    menu.dibujar(ventana)
    pygame.display.flip()
    reloj.tick(60)
