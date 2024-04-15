import pickle
_A=None
_Q='celdas'
_C='celdas.pkl'
_E='felicidad'
_S='ambiente'
_V='atributos'

class Area:
    area_afectada = set()
    area_cubierta = set()
    cords_edificio = set()
    max_radio_afectado = 0 # Funciona para el radio area de Felicidad y Ambiente de los edificios
    max_radio_cubierto = 0 # Funciona para el radio del area cubierta de policia, escuela, bombero y hospital
    x_centro = 0
    y_centro = 0
    max_efecto = 100

    @staticmethod
    def area_defecto(edificio_seleccionado, posicion_edificio):
        if edificio_seleccionado not in ['residencia', 'taller_togas', 'herreria', 'decoracion', 'lecheria', 'refineria', 'policia', 'bombero', 'colegio', 'hospital']:
            posicion_edificio = tuple(posicion_edificio)
            Area.area_afectada.clear()
            Area.area_cubierta.clear()
            Area.area_afectada.add(posicion_edificio)
            Area.area_cubierta.add(posicion_edificio)
            Area.cords_edificio.add(posicion_edificio)
            Area.x_centro, Area.y_centro = posicion_edificio

    @staticmethod
    def area_afectada_(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'residencia' | 'taller_togas' | 'herreria' | 'lecheria' | 'refineria' | 'policia' | 'bombero' | 'colegio' | 'hospital':
                radio = 4
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
        # print(f"El área afectada por {edificio_seleccionado} es: {Area.area_afectada}")
        return list(Area.area_afectada)

    @staticmethod
    def area_afectada_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'decoracion':
                radio = 4
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
        # print(f"El área afectada por {edificio_seleccionado} es: {Area.area_afectada}")
        return list(Area.area_afectada)

    # Se encarga de zona cubierta de servicios
    # lo usaremos también para que no se ubique un edificio al lado de otro
    @staticmethod
    def zona_cubierta_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'policia' | 'bombero' | 'colegio':
                radio = 13
            case 'hospital':
                radio = 11
            case _:
                return
        Area.x_centro, Area.y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        Area.area_cubierta.clear()
        Area.cords_edificio.add(posicion_edificio)
        Area.max_radio_cubierto = radio
        for x in range(max(0, Area.x_centro - radio), min(NUM_CELDAS, Area.x_centro + radio + 1)):
            for y in range(max(0, Area.y_centro - radio), min(NUM_CELDAS, Area.y_centro + radio + 1)):
                if (x - Area.x_centro)**2 + (y - Area.y_centro)**2 <= radio**2:
                    Area.area_cubierta.add((x, y))
        # print(f"La zona cubierta por {edificio_seleccionado} es: {Area.area_cubierta}")
        return list(Area.area_cubierta)


    # Area para Felicidad y Ambiente   
    @staticmethod
    def actualizar_celdas(edificio_seleccionado, edificios):
        if edificio_seleccionado in ['decoracion', 'policia', 'bombero', 'colegio', 'hospital']:
            area_afectada_dict = {coord: _A for coord in Area.area_afectada}
            # print(f"Centro definido en: ({Area.x_centro}, {Area.y_centro})")
            with open(_C, 'rb') as file:
                celdas_data = pickle.load(file)
            for celda_coords in area_afectada_dict:
                celda = next(filter(lambda c: (c['x'], c['y']) == celda_coords, celdas_data[_Q]), _A)
                if celda is not _A:
                    atributos_edificio = edificios[edificio_seleccionado].to_dict()
                    distancia_influencia = ((celda['x'] - Area.x_centro)**2 + (celda['y'] - Area.y_centro)**2)**0.5
                    factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / Area.max_radio_afectado, 1))
                    celda[_V][_E] += round(factor_influencia * atributos_edificio[_E])
                    celda[_V][_S] += round(factor_influencia * atributos_edificio[_S])
            with open(_C, 'wb') as file:
                pickle.dump(celdas_data, file)
            # print(f"Área de {edificio_seleccionado} rellenada")

    # Afecta a la zona que cubre la policia
    @staticmethod
    def servicios_cubiertos(edificio_seleccionado, edificios):
        area_cubierta_dict = {coord: None for coord in Area.area_cubierta}
        with open(_C, 'rb') as file:
            celdas_data = pickle.load(file)

        match edificio_seleccionado:
            case 'policia':
                servicio = 'seguridad'
            case 'bombero':
                servicio = 'incendio'
            case 'colegio':
                servicio = 'educacion'
            case 'hospital':
                servicio = 'salud'
            case _:
                return

        for celda_coords in area_cubierta_dict:
            celda = next(filter(lambda c: (c['x'], c['y']) == celda_coords, celdas_data[_Q]), _A)

            if celda is not None:
                distancia_influencia = ((celda['x'] - Area.x_centro)**2 + (celda['y'] - Area.y_centro)**2)**0.5
                factor_influencia = 0.50 + 0.50 * (1 - min(distancia_influencia / Area.max_radio_cubierto, 1))
                incremento = round(factor_influencia * Area.max_efecto)
                celda['servicios'][servicio] = min(Area.max_efecto, max(celda['servicios'].get(servicio, 0), incremento))

        with open(_C, 'wb') as file:
            pickle.dump(celdas_data, file)
        # print(f"Zona de {edificio_seleccionado} cubierta")