import logging
from areas import Area
from collections import defaultdict
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Condiciones:
    __slots__ = ()

    _N = None
    _T = True
    _F = False
    _D = 'decoracion'
    _A = 'agua'
    _P = 'policia'

    @staticmethod
    def crear_ocupacion(mapa, NUM_CELDAS):
        return defaultdict(lambda: None, {(x, y): mapa[x][y] for x in range(NUM_CELDAS) for y in range(NUM_CELDAS)})

    @staticmethod
    def condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        tamanio_edificio = edificios[edificio_seleccionado].tamanio
        x, y = posicion

        ocupacion = Condiciones.crear_ocupacion(mapa, NUM_CELDAS)

        match edificio_seleccionado:
            case 'suelo':
                if x == 0 or y == 0 or x == NUM_CELDAS - 1 or y == NUM_CELDAS - 1:
                    logger.info("No se puede colocar suelos en los bordes del mapa")
                    return Condiciones._F

                if any(
                    0 <= x + dx < NUM_CELDAS and 0 <= y + dy < NUM_CELDAS and ocupacion[(x + dx, y + dy)] == 'suelo'
                    for dx, dy in itertools.product(range(-2, 3), repeat=2)
                ):
                    logger.info("No se puede colocar suelos cerca de otros suelos")
                    return Condiciones._F

            case _:
                match edificio_seleccionado:
                    case 'suelo' | Condiciones._D | 'lecheria' | 'depuradora' | Condiciones._A | 'bombero':
                        pass
                    case _:
                        if not any(
                            0 <= x + dx < NUM_CELDAS and 0 <= y + dy < NUM_CELDAS and ocupacion[(x + dx, y + dy)] == 'suelo'
                            for dx, dy in itertools.product(range(-1, 2), repeat=2)
                        ):
                            logger.info("Debe ubicarse al lado o en la esquina de un suelo")
                            return Condiciones._F

        if any(
            not (0 <= x + f < NUM_CELDAS and 0 <= y + c < NUM_CELDAS) or ocupacion[(x + f, y + c)] is not Condiciones._N
            for f, c in itertools.product(range(tamanio_edificio[0]), range(tamanio_edificio[1]))
        ):
            return Condiciones._F

        return Condiciones._T

    @staticmethod
    def condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        match edificio_seleccionado:
            case Condiciones._A:
                tiene_vecino_decoracion = any(
                    0 <= posicion[0] + dx < NUM_CELDAS and 0 <= posicion[1] + dy < NUM_CELDAS and mapa[posicion[0] + dx][posicion[1] + dy] == Condiciones._D
                    for dx, dy in itertools.product(range(-1, 3), repeat=2)
                    if not (0 <= dx < 2 and 0 <= dy < 2)
                )

                if not tiene_vecino_decoracion:
                    logger.info("Debe ubicarse al lado o en la esquina de una decoración")
                    return Condiciones._F

                if any(
                    not (0 <= posicion[0] + f < NUM_CELDAS and 0 <= posicion[1] + c < NUM_CELDAS) or mapa[posicion[0] + f][posicion[1] + c] is not None
                    for f, c in itertools.product(range(2), repeat=2)
                ):
                    logger.info("No puede colocarse sobre otro edificio o fuera de los límites del mapa")
                    return Condiciones._F

            case _:
                pass

        return Condiciones._T

    @staticmethod
    def condicion_areas(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        area = Area.area_afectada_por_edificio(edificio_seleccionado, posicion, NUM_CELDAS)
        zona = Area.zona_cubierta_por_edificio(edificio_seleccionado, posicion, NUM_CELDAS)

        if area is None and zona is None:
            return True

        ocupacion = Condiciones.crear_ocupacion(mapa, NUM_CELDAS)

        def verificar_ocupacion(coordenadas):
            return any(
                0 <= x < NUM_CELDAS and 0 <= y < NUM_CELDAS and ocupacion[(x, y)] == edificio_seleccionado
                for x, y in coordenadas
            )

        if area is not None and verificar_ocupacion(area):
            logger.info("Edificio existente en el área.")
            return False

        if zona is not None and verificar_ocupacion(zona):
            logger.info("Edificio existente en la zona.")
            return False

        return True

    @staticmethod
    def condiciones(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
        if not Condiciones.condicion_areas(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios):
            return False
        match edificio_seleccionado:
            case Condiciones._A:
                return Condiciones.condicion_agua(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
            case _:
                return Condiciones.condicion_suelo(edificio_seleccionado, posicion, mapa, NUM_CELDAS, edificios)
