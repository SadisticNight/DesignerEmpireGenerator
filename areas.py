import pickle
_A=None
_Q='celdas'
_C='celdas.pkl'
_E='felicidad'
_S='ambiente'
_V='atributos'
_D='decoracion'
_P='policia'

class Area:
    area_afectada = set()
    area_cubierta = set()
    cords_edificio = set()
    max_radio_afectado = 0 # Funciona para el radio area de Felicidad y Ambiente de los edificios
    max_radio_cubierto = 0 # Funciona para el radio del area cubierta de policia, escuela, bombero y hospital
    x_centro = 0
    y_centro = 0

    @staticmethod
    def area_defecto(edificio_seleccionado, posicion_edificio):
        if edificio_seleccionado not in [_D, _P]:
            posicion_edificio = tuple(posicion_edificio)
            Area.area_afectada.clear()
            Area.area_cubierta.clear()
            Area.area_afectada.add(posicion_edificio)
            Area.area_cubierta.add(posicion_edificio)
            Area.cords_edificio.add(posicion_edificio)
            Area.x_centro, Area.y_centro = posicion_edificio

    @staticmethod
    def area_afectada_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'decoracion':
                radio = 4
            case 'policia':
                radio = 5
            case _:
                return
        Area.x_centro, Area.y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        Area.area_afectada.clear()
        Area.cords_edificio.add(posicion_edificio)
        Area.max_radio_afectado = radio
        for x in range(max(0, Area.x_centro - radio), min(NUM_CELDAS, Area.x_centro + radio + 1)):
            for y in range(max(0, Area.y_centro - radio), min(NUM_CELDAS, Area.y_centro + radio + 1)):
                if (x - Area.x_centro)**2 + (y - Area.y_centro)**2 <= radio**2:
                    Area.area_afectada.add((x, y))
        print(f"El área afectada por {edificio_seleccionado} es: {Area.area_afectada}")
        return list(Area.area_afectada)

    # Se encarga de zona_cubierta
    @staticmethod
    def zona_cubierta_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'policia':
                radio = 13
            case _:
                return
        posicion_edificio = tuple(posicion_edificio)
        Area.area_cubierta.clear()
        Area.cords_edificio.add(posicion_edificio)
        Area.max_radio_cubierto = radio
        for x in range(max(0, Area.x_centro - radio), min(NUM_CELDAS, Area.x_centro + radio + 1)):
            for y in range(max(0, Area.y_centro - radio), min(NUM_CELDAS, Area.y_centro + radio + 1)):
                if (x - Area.x_centro)**2 + (y - Area.y_centro)**2 <= radio**2:
                    Area.area_cubierta.add((x, y))
        print(f"La zona cubierta por {edificio_seleccionado} es: {Area.area_cubierta}")
        print(f"El máximo rádio cubierto es: {Area.max_radio_cubierto}")
        return list(Area.area_cubierta)

    # Area para Felicidad y Ambiente   
    @staticmethod
    def actualizar_celdas(edificio_seleccionado, edificios):
        if edificio_seleccionado in [_D, _P]:
            area_afectada_dict = {coord: _A for coord in Area.area_afectada}
            # print(f"Centro definido en: ({Area.x_centro}, {Area.y_centro})")
            with open(_C, 'rb') as file:
                celdas_data = pickle.load(file)
            for celda_coords in area_afectada_dict:
                celda = next((c for c in celdas_data[_Q] if (c['x'], c['y']) == celda_coords), _A)
                if celda is not _A:
                    atributos_edificio = edificios[edificio_seleccionado].to_dict()
                    distancia_influencia = ((celda['x'] - Area.x_centro)**2 + (celda['y'] - Area.y_centro)**2)**0.5
                    # factor_influencia = 1.00 - 0.50 * (distancia_influencia / Area.max_radio_afectado)
                    factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / Area.max_radio_afectado, 1))
                    celda[_V][_E] += round(factor_influencia * atributos_edificio[_E])
                    celda[_V][_S] += round(factor_influencia * atributos_edificio[_S])
            with open(_C, 'wb') as file:
                pickle.dump(celdas_data, file)
            print(f"Área de {edificio_seleccionado} rellenada")

    # Afecta a la zona que cubre la policia
    @staticmethod
    def servicios_cubiertos(edificio_seleccionado, edificios):
        area_cubierta_dict = {coord: None for coord in Area.area_cubierta}
        # print(f"Centro definido en: ({Area.x_centro}, {Area.y_centro})")
        match edificio_seleccionado:
            case 'policia':
                with open(_C, 'rb') as file:
                    celdas_data = pickle.load(file)

                for celda_coords in area_cubierta_dict:
                    celda = next((c for c in celdas_data[_Q] if (c['x'], c['y']) == celda_coords), None)
                    if celda is not None:
                        distancia_influencia = ((celda['x'] - Area.x_centro)**2 + (celda['y'] - Area.y_centro)**2)**0.5
                        factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / Area.max_radio_cubierto, 1))
                        incremento = round(factor_influencia * 100)
                        celda['servicios']['seguridad'] = min(100, max(celda['servicios'].get('seguridad', 0), incremento))
                with open(_C, 'wb') as file:
                    pickle.dump(celdas_data, file)
                print(f"Zona de policía cubierta")

            case _:
                True