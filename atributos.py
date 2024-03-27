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
    def __init__(self, color, energia, agua, basura, comida, empleos, residentes, tipo, felicidad, ambiente, tamanio):
        # Se ajusta la validación de 'color' para aceptar listas de enteros y convertirlas a tupla.
        if isinstance(color, list) and all(isinstance(c, int) for c in color):
            color = tuple(color)
        elif not isinstance(color, tuple) or not all(isinstance(c, int) for c in color):
            raise ValueError("El color debe ser una tupla o lista de enteros")
        
        self.color = color
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
        return {
            "color": self.color,
            "energia": self.energia,
            "agua": self.agua,
            "basura": self.basura,
            "comida": self.comida,
            "empleos": self.empleos,
            "residentes": self.residentes,
            "tipo": self.tipo,
            "felicidad": self.felicidad,
            "ambiente": self.ambiente,
            "tamanio": self.tamanio,
        }