@0xafdafdafdafdafda;

struct Edificio {
  nombre @0 :Text;             # Nombre único del edificio
  color @1 :List(UInt8);       # Valores RGB (0-255)
  energia @2 :Int32;           # Energía (puede ser negativo)
  agua @3 :Int32;              # Agua (puede ser negativo)
  basura @4 :Int32;            # Basura (puede ser negativo o positivo)
  comida @5 :Int32;            # Comida (puede ser negativo o positivo)
  empleos @6 :Int32;           # Número de empleos
  residentes @7 :Int32;        # Número de residentes
  tipo @8 :Text;               # Tipo de edificio (por ejemplo, "residencia", "industria", etc.)
  felicidad @9 :Int32;         # Índice de felicidad
  ambiente @10 :Int32;         # Calidad ambiental
  tamanio @11 :List(UInt16);   # Tamaño (ancho, alto)
}

struct Edificios {
  lista @0 :List(Edificio);    # Lista de edificios
}

const edificiosPredefinidos :Edificios = (
  lista = [
    (nombre = "suelo", color = [128, 128, 128], energia = 0,   agua = 0,    basura = 0,    comida = 0,    empleos = 0,    residentes = 0,   tipo = "suelo",     felicidad = 0,   ambiente = 0,    tamanio = [1, 1]),
    (nombre = "residencia", color = [255, 20, 147], energia = -77,  agua = -124, basura = -50,  comida = -57,  empleos = 0,    residentes = 3092, tipo = "residencia", felicidad = -93, ambiente = -55,  tamanio = [1, 1]),
    (nombre = "taller_togas", color = [0, 0, 255], energia = -95,  agua = -83,  basura = -61,  comida = -70,  empleos = 1891, residentes = 0,   tipo = "comercio",   felicidad = -121,ambiente = -83,  tamanio = [1, 1]),
    (nombre = "herreria", color = [255, 255, 0], energia = -48,  agua = -42,  basura = -33,  comida = -36,  empleos = 2613, residentes = 0,   tipo = "industria", felicidad = -57, ambiente = -42,  tamanio = [1, 1]),
    (nombre = "decoracion", color = [0, 100, 0], energia = 0,    agua = 0,    basura = 0,    comida = 0,    empleos = 0,    residentes = 0,   tipo = "decoracion", felicidad = 6600, ambiente = 10000,tamanio = [1, 1]),
    (nombre = "lecheria", color = [165, 42, 42], energia = -30,  agua = -32,  basura = -23,  comida = 4000, empleos = 165,  residentes = 0,   tipo = "industria", felicidad = 1022,ambiente = 22,   tamanio = [1, 1]),
    (nombre = "depuradora", color = [0, 0, 0], energia = -60,   agua = -70,  basura = 40000, comida = -60,  empleos = 465,  residentes = 0,   tipo = "industria", felicidad = -118,ambiente = -119, tamanio = [2, 2]),
    (nombre = "agua", color = [0, 255, 255], energia = -60,  agua = 40000, basura = -40,  comida = -60,  empleos = 503,  residentes = 0,   tipo = "industria", felicidad = -25, ambiente = -25,  tamanio = [2, 2]),
    (nombre = "refineria", color = [255, 0, 255], energia = 10000, agua = -20, basura = -40,  comida = -23,  empleos = 124,  residentes = 0,   tipo = "industria", felicidad = -23, ambiente = -22,  tamanio = [1, 1]),
    (nombre = "policia", color = [0, 0, 128], energia = -68,  agua = -62,  basura = -68,  comida = -49,  empleos = 216,  residentes = 0,   tipo = "comercio",   felicidad = 1153,ambiente = -80,  tamanio = [1, 1]),
    (nombre = "bombero", color = [255, 0, 0], energia = -43,   agua = -40,  basura = -43,  comida = -32,  empleos = 126,  residentes = 0,   tipo = "comercio",   felicidad = 672, ambiente = -75,  tamanio = [1, 1]),
    (nombre = "colegio", color = [128, 0, 128], energia = -35,  agua = -32,  basura = -35,  comida = -27,  empleos = 96,   residentes = 0,   tipo = "comercio",   felicidad = 513, ambiente = -40,  tamanio = [1, 1]),
    (nombre = "hospital", color = [255, 165, 0], energia = -14,  agua = -14,  basura = -14,  comida = -13,  empleos = 16,   residentes = 0,   tipo = "comercio",   felicidad = 87,  ambiente = -15,  tamanio = [1, 1])
  ]
);
