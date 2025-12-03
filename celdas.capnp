@0xabcdefabcdefabcdef;

struct Celda {
  x @0 :UInt64;       # Coordenada X (64 bits)
  y @1 :UInt64;       # Coordenada Y (64 bits)
  hash @2 :UInt64;      # Hash único para la celda
  edificio @3 :Text;  # Tipo de edificio
  tipo @4 :Text;      # Tipo de celda
  atributos @5 :Atributos;
  servicios @6 :Servicios;
}

struct Atributos {
  energia @0 :Int64;  # Energía disponible
  agua @1 :Int64;     # Agua disponible
  basura @2 :Int64;   # Capacidad de manejo de basura
  comida @3 :Int64;   # Comida producida
  empleos @4 :Int64;  # Empleos disponibles
  residentes @5 :Int64;  # Número de residentes
  felicidad @6 :Int64;   # Índice de felicidad
  ambiente @7 :Int64;    # Calidad ambiental
}

struct Servicios {
  seguridad @0 :Int64;  # Nivel de seguridad
  incendio @1 :Int64;   # Nivel de protección contra incendios
  salud @2 :Int64;      # Calidad del sistema de salud
  educacion @3 :Int64;  # Nivel educativo
}

struct Mapa {
  celdas @0 :List(Celda);
}