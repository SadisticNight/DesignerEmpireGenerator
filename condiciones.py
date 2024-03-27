def verificar_decoracion(ubicaciones):
    """
    Verifica si hay al menos un edificio de decoración en el mapa.
    :param ubicaciones: Lista o diccionario con las ubicaciones de los edificios actuales.
    :return: True si hay un edificio de decoración, False en caso contrario.
    """
    for edificio in ubicaciones:
        if edificio['tipo'] == 'decoración':
            return True
    return False
