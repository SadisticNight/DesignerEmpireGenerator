# edificios.py
from enum import Enum
from atributos import Atributo
import capnp

class TipoEdificio(Enum):
    COMERCIO="comercio"; INDUSTRIA="industria"; RESIDENCIA="residencia"; DECORACION="decoracion"; SUELO="suelo"

# toggle de log en arranque (True para ver la lista en consola)
SHOW_LOAD=True

# schema + bin
ec=capnp.load("edificios.capnp")
with open("edificios.bin","rb") as f: m=ec.Edificios.read(f)

# atajos locales
A=Atributo; T=TipoEdificio; it=int

# diccionario principal (clave = nombre)
edificios={
    e.nombre: A(
        color=tuple(map(it,e.color)),
        energia=e.energia, agua=e.agua, basura=e.basura, comida=e.comida,
        empleos=e.empleos, residentes=e.residentes,
        tipo=T[e.tipo.upper()],
        felicidad=e.felicidad, ambiente=e.ambiente,
        tamanio=tuple(map(it,e.tamanio))
    ) for e in m.lista
}

# log opcional
if SHOW_LOAD:
    print("Edificios cargados:")
    for e in m.lista:
        print(f"  nombre: {e.nombre}, tipo: {e.tipo}, color: {[it(c) for c in e.color]}")
