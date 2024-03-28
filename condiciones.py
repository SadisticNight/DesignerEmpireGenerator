_N=None
_T=True
_F=False
_D='decoracion'
_A='agua'

def condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
    tamanio_edificio = edificios[edificio_seleccionado].tamanio
    x, y = posicion

    # Verificar si el edificio es 'suelo' y si está en el borde del mapa
    if edificio_seleccionado == 'suelo':
        if x == 0 or y == 0 or x == NUM_CELDAS - 1 or y == NUM_CELDAS - 1:
            print("No se puede colocar suelos en los bordes del mapa")
            return _F

        # Verificar si hay edificios de tipo 'suelo' en las celdas vecinas
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not _N and mapa[nx][ny] == 'suelo':
                    print("No se puede colocar suelos cerca de otros suelos")
                    return _F
    else:
        # Para todos los edificios excepto decoración, lechería, depuradora, agua, bombero
        if edificio_seleccionado not in ['suelo', _D, 'lecheria', 'depuradora', _A, 'bombero']:
            # Verificar si hay al menos un edificio de tipo 'suelo' en las celdas vecinas
            tiene_vecino_suelo = _F
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not _N and mapa[nx][ny] == 'suelo':
                        tiene_vecino_suelo = _T
                        break
                if tiene_vecino_suelo:
                    break
            if not tiene_vecino_suelo:
                print("Debe ubicarse al lado o en la esquina de un suelo")
                return _F
        else:
            pass

    # Verificar si el espacio está disponible
    for f in range(tamanio_edificio[0]):
        for c in range(tamanio_edificio[1]):
            if mapa[x + f][y + c] is not _N:
                return _F

    return _T

def condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
    x, y = posicion

    # Verificar si el edificio es _A
    if edificio_seleccionado == _A:
        # Verificar si hay al menos un edificio de tipo _D en las celdas vecinas
        tiene_vecino_decoracion = _F
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not _N and mapa[nx][ny] == _D:
                    tiene_vecino_decoracion = _T
                    break
            if tiene_vecino_decoracion:
                break
        if not tiene_vecino_decoracion:
            print("Debe ubicarse al lado o en la esquina de una decoración")
            return _F
    else:
        pass

    return _T

def condiciones(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
    if edificio_seleccionado == _A:
        return condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
    else:
        return condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
