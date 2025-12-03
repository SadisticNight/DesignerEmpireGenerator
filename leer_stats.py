import capnp; sC=capnp.load("stats.capnp")  # Carga el esquema Stats

# Lista de nombres predefinidos (la misma que se usó en generar_stats.py)
ne=["suelo","residencia","taller_togas","herreria","decoracion","lecheria",
    "depuradora","agua","refineria","policia","bombero","colegio","hospital"]

def ls(fn):  # Lee stats desde archivo binario
    with open(fn,"rb") as f: return sC.Stats.read(f)

def ps(s):  # Imprime stats con etiquetas claras
    print("Total Residentes:", s.totalResidentes,
          "| Total Empleos Industria:", s.totalEmpleosIndustria,
          "| Total Empleos Comercio:", s.totalEmpleosComercio,
          "| Total Empleos:", s.totalEmpleos)
    print("Porcentaje Industria:", s.porcentajeIndustria,
          "| Porcentaje Comercio:", s.porcentajeComercio,
          "| Desequilibrio Laboral:", s.desequilibrioLaboral,
          "| EC:", s.ec)
    print("Energia Usada:", s.energiaUsada,
          "| Energia Total:", s.energiaTotal,
          "| Agua Usada:", s.aguaUsada,
          "| Agua Total:", s.aguaTotal)
    print("Comida Usada:", s.comidaUsada,
          "| Comida Total:", s.comidaTotal,
          "| Basura Usada:", s.basuraUsada,
          "| Basura Total:", s.basuraTotal)
    print("Ecologia Total:", s.ecologiaTotal,
          "| Felicidad Total:", s.felicidadTotal)
    # Usa la lista 'ne' para mostrar nombres de edificio, ya que e.nombre puede estar vacio
    for i, e in enumerate(s.cantidadEdificios):
        print(f"{ne[i]}: {e.cantidad}")

if __name__=="__main__":  # Bloque principal: lee y muestra stats
    ps(ls("estadisticas.bin"))
