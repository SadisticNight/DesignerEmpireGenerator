import pickle

_D='celdas'
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

    @staticmethod
    def area_defecto(edificio_seleccionado, posicion_edificio):
        if edificio_seleccionado not in ['decoracion', 'policia']:
            posicion_edificio = tuple(posicion_edificio)
            Area.area_afectada.clear()
            Area.area_cubierta.clear()
            Area.area_afectada.add(posicion_edificio)
            Area.area_cubierta.add(posicion_edificio)
            Area.cords_edificio.add(posicion_edificio)


    @staticmethod
    def area_afectada_por_edificio(edificio_seleccionado, posicion_edificio, NUM_CELDAS=200):
        match edificio_seleccionado:
            case 'decoracion':
                radio = 4
            case 'policia':
                radio = 5
            case _:
                return

        x_centro, y_centro = posicion_edificio
        posicion_edificio = tuple(posicion_edificio)
        Area.area_afectada.clear()
        Area.cords_edificio.add(posicion_edificio)
        Area.max_radio_afectado = radio

        for x in range(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1)):
            for y in range(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1)):
                if (x - x_centro)**2 + (y - y_centro)**2 <= radio**2:
                    Area.area_afectada.add((x, y))

        print(f"El área afectada por {edificio_seleccionado} es: {Area.area_afectada}")
        return list(Area.area_afectada)

    
    # Se encarga de zona_cubierta
    @staticmethod
    def zona_policia(edificio_seleccionado, posicion_edificio, radio=13, NUM_CELDAS=200):
        if edificio_seleccionado in ['policia']:
            x_centro, y_centro = posicion_edificio
            posicion_edificio = tuple(posicion_edificio)
            Area.area_cubierta.clear()
            Area.cords_edificio.add(posicion_edificio)
            Area.max_radio_cubierto = radio

            for x in range(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1)):
                for y in range(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1)):
                    if (x - x_centro)**2 + (y - y_centro)**2 <= radio**2:
                        Area.area_cubierta.add((x, y))
                        
            print(f"El área cubierta es: {Area.area_cubierta}")
            print(f"El máximo rádio cubierto es: {Area.max_radio_cubierto}")
            return list(Area.area_cubierta)
    

    # Area para Felicidad y Ambiente   
    @staticmethod
    def actualizar_celdas(edificio_seleccionado, edificios):
        if edificio_seleccionado in ['decoracion', 'policia']:
            area_afectada_dict = {coord: None for coord in Area.area_afectada}
            x_centro, y_centro = list(Area.cords_edificio)[0]
            print(f"La distancia máxima es: {Area.max_radio_afectado}")

            with open(_C, 'rb') as file:
                celdas_data = pickle.load(file)

            for celda_coords in area_afectada_dict:
                celda = next((c for c in celdas_data[_D] if (c['x'], c['y']) == celda_coords), None)
                if celda is not None:
                    atributos_edificio = edificios[edificio_seleccionado].to_dict()
                    distancia_influencia = ((celda['x'] - x_centro)**2 + (celda['y'] - y_centro)**2)**0.5
                    factor_influencia = 1.00 - 0.50 * (distancia_influencia / Area.max_radio_afectado)
                    print(f"Celda {celda_coords}: Factor de Influencia = {factor_influencia:.4f}")
                    celda[_V][_E] += factor_influencia * atributos_edificio[_E]
                    celda[_V][_S] += factor_influencia * atributos_edificio[_S]
                    print(f"Celda {celda_coords}: Felicidad Final = {celda[_V][_E]}, Ambiente Final = {celda[_V][_S]}")

            with open(_C, 'wb') as file:
                pickle.dump(celdas_data, file)
            print(f"Área de {edificio_seleccionado} rellenada")



    # Afecta a la zona que cubre la policia
    @staticmethod
    def servicios_cubiertos(edificio_seleccionado, edificios):
        area_cubierta_dict = {coord: None for coord in Area.area_cubierta}
        x_centro, y_centro = list(Area.cords_edificio)[0]

        match edificio_seleccionado:
            case 'policia':
                with open(_C, 'rb') as file:
                    celdas_data = pickle.load(file)
                for celda_coords in area_cubierta_dict:
                    celda = next((c for c in celdas_data[_D] if (c['x'], c['y']) == celda_coords), None)
                    if celda is not None:
                        servicios_edificio = edificios[edificio_seleccionado].to_dict()
                        distancia_influencia = ((celda['x'] - x_centro)**2 + (celda['y'] - y_centro)**2)**0.5
                        factor_influencia = 1.00 - 0.50 * (distancia_influencia / Area.max_radio_cubierto)
                        incremento = factor_influencia * servicios_edificio.get('seguridad', 0)
                        celda['servicios']['seguridad'] = min(100, celda['servicios']['seguridad'] + incremento)
                with open(_C, 'wb') as file:
                    pickle.dump(celdas_data, file)
                print(f"Zona de policía cubierta")
            
            case _:
                True