def condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
    tamanio_edificio = edificios[edificio_seleccionado].tamanio
    x, y = posicion

    # Verificar si el edificio es 'suelo' y si está en el borde del mapa
    if edificio_seleccionado == 'suelo':
        if x == 0 or y == 0 or x == NUM_CELDAS - 1 or y == NUM_CELDAS - 1:
            print("No se puede colocar suelos en los bordes del mapa")
            return False

        # Verificar si hay edificios de tipo 'suelo' en las celdas vecinas
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not None and mapa[nx][ny] == 'suelo':
                    print("No se puede colocar suelos cerca de otros suelos")
                    return False
    else:
        # Para todos los edificios excepto decoración, lechería, depuradora, agua, bombero
        if edificio_seleccionado not in ['suelo', 'decoracion', 'lecheria', 'depuradora', 'agua', 'bombero']:
            # Verificar si hay al menos un edificio de tipo 'suelo' en las celdas vecinas
            tiene_vecino_suelo = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not None and mapa[nx][ny] == 'suelo':
                        tiene_vecino_suelo = True
                        break
                if tiene_vecino_suelo:
                    break
            if not tiene_vecino_suelo:
                print("Debe ubicarse al lado o en la esquina de un suelo")
                return False
        else:
            pass

    # Verificar si el espacio está disponible
    for f in range(tamanio_edificio[0]):
        for c in range(tamanio_edificio[1]):
            if mapa[x + f][y + c] is not None:
                return False

    return True

def condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
    x, y = posicion

    # Verificar si el edificio es 'agua'
    if edificio_seleccionado == 'agua':
        # Verificar si hay al menos un edificio de tipo 'decoracion' en las celdas vecinas
        tiene_vecino_decoracion = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not None and mapa[nx][ny] == 'decoracion':
                    tiene_vecino_decoracion = True
                    break
            if tiene_vecino_decoracion:
                break
        if not tiene_vecino_decoracion:
            print("Debe ubicarse al lado o en la esquina de una decoración")
            return False
    else:
        pass

    return True

def condiciones(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
    if edificio_seleccionado == 'agua':
        return condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
    else:
        return condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
