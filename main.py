import pygame,sys,numpy as np
from menu import Menu
from edificios import edificios
from atributos import Atributo
from generar_pickle import generar_datos
import os,pickle,hashlib,time
from condiciones import condicion_suelo

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
_F='No se puede colocar el edificio aquí'
_Y='edificio'
_D='celdas'
_H='hash'
_C='celdas.pkl'
_A='energia'
_T='agua'
_R='basura'
_I='comida'
_B='empleos'
_U='residentes'
_E='felicidad'
_S='ambiente'
_V='atributos'
_Z='tipo'

if not os.path.exists(_C):generar_datos()

pygame.init()
ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
pygame.display.set_caption('Juego de Construcción de Ciudades')
reloj = pygame.time.Clock()
menu = Menu(ANCHO_VENTANA, ALTO_VENTANA)
posicion_usuario = [NUM_CELDAS // 2, NUM_CELDAS // 2]
ultima_posicion = list(posicion_usuario)
mapa = np.full((NUM_CELDAS, NUM_CELDAS), _N)
edificio_seleccionado = _N

def generar_hash(edificio):
    return hashlib.sha256((edificio + str(time.time())).encode()).hexdigest()

def actualizar_seleccion():
    global tamanio_cursor, edificio_seleccionado
    if edificio_seleccionado is not None:
        tamanio_cursor = edificios[edificio_seleccionado].tamanio
    else:
        tamanio_cursor = (1, 1)

def dibujar_celda(fila, columna, atributo):
    tamanio_edificio = atributo.tamanio
    if atributo.color == VERDE:  # Si el atributo es el cursor
        rect = pygame.Rect(columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA * tamanio_edificio[0], TAMANO_CELDA * tamanio_edificio[1])
    else:  # Si el atributo es un edificio
        if tamanio_edificio == [2, 2]:
            rect = pygame.Rect(columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA * (tamanio_edificio[0] - 1), TAMANO_CELDA * (tamanio_edificio[1] - 1))
        if tamanio_edificio == [1, 1]:
            rect = pygame.Rect(columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA * tamanio_edificio[0], TAMANO_CELDA * tamanio_edificio[1])
    pygame.draw.rect(ventana, atributo.color, rect, 0)


def dibujar_cuadricula():
    global tamanio_cursor
    ventana.fill(BLANCO)
    actualizar_seleccion()
    
    # Dibujar todos los edificios primero
    for fila in range(NUM_CELDAS):
        for columna in range(NUM_CELDAS):
            if mapa[fila][columna] != _N:
                atributo_edificio = edificios[mapa[fila][columna]]
                dibujar_celda(fila, columna, atributo_edificio)

    # Dibujar el cursor del usuario
    dibujar_celda(posicion_usuario[0], posicion_usuario[1], Atributo(VERDE, 0, 0, 0, 0, 0, 0, None, 0, 0, list(tamanio_cursor)))

def actualizar_pantalla():
    dibujar_cuadricula()
    menu.dibujar(ventana)
    pygame.display.flip()

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
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_n:
                x, y = posicion_usuario
                if edificio_seleccionado is not _N:
                    tamanio_edificio = edificios[edificio_seleccionado].tamanio
                    if tamanio_edificio == [2, 2]:
                        print(f"Coordenadas 2x2: ({x}, {y}), ({x+1}, {y}), ({x}, {y+1}), ({x+1}, {y+1})")
                    else:
                        print(f"Coordenada 1x1: ({x}, {y})")
                else:
                    print(f"Coordenada inicial 1x1: ({x}, {y})")

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
                if edificio_seleccionado is _N:
                    print('Debe seleccionar un edificio')
                else:
                    if condicion_suelo(edificio_seleccionado, posicion_usuario, mapa, NUM_CELDAS, edificios):
                        tamanio_edificio = edificios[edificio_seleccionado].tamanio
                        if tamanio_edificio == [2, 2]:
                            if posicion_usuario[0] + 1 >= NUM_CELDAS or posicion_usuario[1] + 1 >= NUM_CELDAS:
                                print(_F)
                            else:
                                puede_colocar = True
                                for f in range(tamanio_edificio[0]):
                                    for c in range(tamanio_edificio[1]):
                                        if mapa[posicion_usuario[0] + f][posicion_usuario[1] + c] is not _N:
                                            puede_colocar = False
                                            break
                                    if not puede_colocar:
                                        break
                                if puede_colocar:
                                    hash_edificio = generar_hash(edificio_seleccionado)
                                    atributos_edificio = edificios[edificio_seleccionado].to_dict()
                                    for f in range(tamanio_edificio[0]):
                                        for c in range(tamanio_edificio[1]):
                                            mapa[posicion_usuario[0] + f][posicion_usuario[1] + c] = edificio_seleccionado
                                            with open(_C, 'rb') as file:
                                                celdas_data = pickle.load(file)
                                            for celda in celdas_data[_D]:
                                                if celda['x'] == posicion_usuario[0] + f and celda['y'] == posicion_usuario[1] + c:
                                                    celda[_Y] = edificio_seleccionado
                                                    celda[_H] = hash_edificio
                                                    celda[_Z] = atributos_edificio[_Z].value
                                                    celda[_V] = {key: atributos_edificio[key] for key in [_A, _T, _R, _I, _B, _U, _E, _S]}
                                                    break
                                            with open(_C, 'wb') as file:
                                                pickle.dump(celdas_data, file)
                                    print(f"Edificio {edificio_seleccionado} colocado en las coordenadas: ({posicion_usuario[0]}, {posicion_usuario[1]}), ({posicion_usuario[0]+1}, {posicion_usuario[1]}), ({posicion_usuario[0]}, {posicion_usuario[1]+1}), ({posicion_usuario[0]+1}, {posicion_usuario[1]+1})")
                                else:
                                    print(_F)
                        else:
                            # Para edificios de tamaño 1x1, aplicar la lógica anterior
                            puede_colocar = True
                            if mapa[posicion_usuario[0]][posicion_usuario[1]] is not _N:
                                puede_colocar = False
                            if puede_colocar:
                                hash_edificio = generar_hash(edificio_seleccionado)
                                atributos_edificio = edificios[edificio_seleccionado].to_dict()  # Obtén los atributos del edificio
                                mapa[posicion_usuario[0]][posicion_usuario[1]] = edificio_seleccionado
                                with open(_C, 'rb') as file:
                                    celdas_data = pickle.load(file)
                                for celda in celdas_data[_D]:
                                    if celda['x'] == posicion_usuario[0] and celda['y'] == posicion_usuario[1]:
                                        celda[_Y] = edificio_seleccionado
                                        celda[_H] = hash_edificio
                                        celda[_Z] = atributos_edificio[_Z].value
                                        celda[_V] = {key: atributos_edificio[key] for key in [_A, _T, _R, _I, _B, _U, _E, _S]} 
                                        break
                                with open(_C, 'wb') as file:
                                    pickle.dump(celdas_data, file)
                                print(f"Edificio {edificio_seleccionado} colocado en {posicion_usuario}")
                            else:
                                print(_F)

                actualizar_pantalla()

            if evento.key in (pygame.K_KP_MINUS, pygame.K_MINUS):
                if tamanio_cursor == [1, 1]:
                    with open(_C, 'rb') as file:
                        celdas_data = pickle.load(file)
                    for celda in celdas_data[_D]:
                        if celda['x'] == posicion_usuario[0] and celda['y'] == posicion_usuario[1]:
                            if celda[_Y] != '':
                                hash_a_eliminar = celda[_H]
                                atributos_edificio = edificios[celda[_Y]].to_dict()
                                break
                    else:
                        print('No hay edificio en esta celda')
                        continue

                    for celda in celdas_data[_D]:
                        if celda[_H] == hash_a_eliminar:
                            celda[_Y] = ''
                            celda[_H] = ''
                            celda[_Z] = ''  
                            celda[_V] = {key: celda[_V][key] - atributos_edificio[key] for key in [_A, _T, _R, _I, _B, _U, _E, _S]} 
                            mapa[celda['x']][celda['y']] = _N 
                    with open(_C, 'wb') as file:
                        pickle.dump(celdas_data, file)
                    print('Edificio eliminado')
                else:
                    print('El cursor debe ser 1x1')
                actualizar_pantalla()


            if menu.menu_activo:
                menu.manejar_evento(evento)

    menu.dibujar(ventana)
    pygame.display.flip()
    reloj.tick(60)
