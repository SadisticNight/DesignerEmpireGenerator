# generar_stats.py
import capnp; sC=capnp.load("stats.capnp")  # → Carga el esquema Stats de Cap'n Proto
ne=["suelo","residencia","taller_togas","herreria","decoracion","lecheria","depuradora","agua","refineria","policia","bombero","colegio","hospital"]  # → Nombres predefinidos

def generar_stats():  # → Genera mensaje Stats con valores iniciales
    m=sC.Stats.new_message()
    # iniciales en cero/False (ruta caliente compacta, sin comprensiones-side effects)
    m.totalResidentes=m.totalEmpleosIndustria=m.totalEmpleosComercio=m.totalEmpleos=0
    m.porcentajeIndustria=m.porcentajeComercio=0.0
    m.desequilibrioLaboral=0; m.ec=False
    m.energiaUsada=m.energiaTotal=m.aguaUsada=m.aguaTotal=0
    m.comidaUsada=m.comidaTotal=m.basuraUsada=m.basuraTotal=0
    m.ecologiaTotal=m.felicidadTotal=0
    lst=m.init("cantidadEdificios",len(ne))
    for i,n in enumerate(ne):
        e=lst[i]; e.nombre=n; e.cantidad=0
    return m

if __name__=="__main__":
    with open("estadisticas.bin","wb") as f:
        generar_stats().write(f)
    print("Archivo estadisticas.bin generado correctamente.")
