class Atributo:
    """
    Representa los atributos de un edificio, incluyendo aspectos como color, recursos consumidos o producidos, y tamaño.

    Args:
    color (tuple): Representa el color RGB del edificio.
    energia, agua, basura, comida (int): Representan los recursos consumidos o producidos.
    empleos, residentes (int): Indican la cantidad de empleos disponibles y la capacidad residencial.
    tipo (str): El tipo del edificio (comercio, industria, etc.).
    felicidad, ambiente (int): Impacto del edificio en la felicidad y el ambiente.
    tamaño (tuple): Dimensiones del edificio (ancho, largo).
    """
    __slots__ = ['color', 'energia', 'agua', 'basura', 'comida', 'empleos', 'residentes', 'tipo', 'felicidad', 'ambiente', 'tamanio']

    def __init__(self, color, energia, agua, basura, comida, empleos, residentes, tipo, felicidad, ambiente, tamanio):
        if isinstance(color, (list, tuple)) and all(isinstance(c, int) for c in color):
            self.color = tuple(color)
        else:
            raise ValueError("El color debe ser una tupla o lista de enteros")
        
        self.energia = energia
        self.agua = agua
        self.basura = basura
        self.comida = comida
        self.empleos = empleos
        self.residentes = residentes
        self.tipo = tipo
        self.felicidad = felicidad
        self.ambiente = ambiente
        self.tamanio = tamanio

    def to_dict(self):
        """
        Convierte los atributos del edificio a un diccionario.

        Returns:
        dict: Diccionario con los atributos del edificio.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}
