from areas import Area

class Condiciones:

    _N=None
    _T=True
    _F=False
    _D='decoracion'
    _A='agua'
    _P='policia'

    @staticmethod
    def condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        tamanio_edificio = edificios[edificio_seleccionado].tamanio
        x, y = posicion

        # Verificar si el edificio es 'suelo' y si está en el borde del mapa
        if edificio_seleccionado == 'suelo':
            if x == 0 or y == 0 or x == NUM_CELDAS - 1 or y == NUM_CELDAS - 1:
                print("No se puede colocar suelos en los bordes del mapa")
                return Condiciones._F

            # Verificar si hay edificios de tipo 'suelo' en las celdas vecinas
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not Condiciones._N and mapa[nx][ny] == 'suelo':
                        print("No se puede colocar suelos cerca de otros suelos")
                        return Condiciones._F
        else:
            # Para todos los edificios excepto decoración, lechería, depuradora, agua, bombero
            if edificio_seleccionado not in ['suelo', Condiciones._D, 'lecheria', 'depuradora', Condiciones._A, 'bombero']:
                # Verificar si hay al menos un edificio de tipo 'suelo' en las celdas vecinas
                tiene_vecino_suelo = Condiciones._F
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not Condiciones._N and mapa[nx][ny] == 'suelo':
                            tiene_vecino_suelo = Condiciones._T
                            break
                    if tiene_vecino_suelo:
                        break
                if not tiene_vecino_suelo:
                    print("Debe ubicarse al lado o en la esquina de un suelo")
                    return Condiciones._F
            else:
                pass

        # Verificar si el espacio está disponible
        for f in range(tamanio_edificio[0]):
            for c in range(tamanio_edificio[1]):
                if mapa[x + f][y + c] is not Condiciones._N:
                    return Condiciones._F

        return Condiciones._T
    
    @staticmethod
    def condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        x, y = posicion

        # Verificar si el edificio es Condiciones._A
        if edificio_seleccionado == Condiciones._A:
            # Verificar si hay al menos un edificio de tipo Condiciones._D en las celdas vecinas
            tiene_vecino_decoracion = Condiciones._F
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < NUM_CELDAS and 0 <= ny < NUM_CELDAS and mapa[nx][ny] is not Condiciones._N and mapa[nx][ny] == Condiciones._D:
                        tiene_vecino_decoracion = Condiciones._T
                        break
                if tiene_vecino_decoracion:
                    break
            if not tiene_vecino_decoracion:
                print("Debe ubicarse al lado o en la esquina de una decoración")
                return Condiciones._F
        else:
            pass

        return Condiciones._T
    
    @staticmethod
    def condicion_areas(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        area = Area.area_afectada_por_edificio(edificio_seleccionado, posicion, NUM_CELDAS)
        
        if area is None:
            return True  # Si la función no devuelve un área, asumimos que no necesita comprobación

        for x, y in area:
            if 0 <= x < NUM_CELDAS and 0 <= y < NUM_CELDAS and mapa[x][y] == edificio_seleccionado:
                print("No se puede colocar el edificio aquí porque ya hay un edificio del mismo tipo en el área.")
                return False

        return True


    @staticmethod
    def condiciones(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        if not Condiciones.condicion_areas(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
            return False
        if edificio_seleccionado == Condiciones._A:
            return Condiciones.condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
        else:
            return Condiciones.condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)