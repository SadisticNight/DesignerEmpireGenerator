# generar_celdas.py
# → Genera celdas.bin (Cap'n Proto) con un mapa BOARD_SIZE x BOARD_SIZE sin usar NumPy
import capnp
from config import BOARD_SIZE as N  # fuente única de verdad
c = capnp.load("celdas.capnp")

def generar_datos():
    # → Mensaje y tabla de celdas
    T = N * N
    m = c.Mapa.new_message()
    cs = m.init("celdas", T)

    # → Asignación masiva sin comprensiones con efectos (evita lista temporal) ni lambdas
    #    Minimiza overhead: cache de atributos, servicios y funciones builtin
    rng = range
    div = divmod
    for i in rng(T):
        x, y = div(i, N)
        cel = cs[i]
        cel.x = x; cel.y = y; cel.hash = ""; cel.edificio = ""; cel.tipo = ""
        at = cel.atributos; sv = cel.servicios
        # atributos
        at.energia = 0; at.agua = 0; at.basura = 0; at.comida = 0
        at.empleos = 0; at.residentes = 0; at.felicidad = 0; at.ambiente = 0
        # servicios
        sv.seguridad = 0; sv.incendio = 0; sv.salud = 0; sv.educacion = 0
    return m

if __name__ == "__main__":
    datos_generados = generar_datos()
    with open("celdas.bin", "wb") as f:
        datos_generados.write(f)
    print(f"Cap'n Proto generado correctamente con {N} x {N} = {N*N} celdas.")
