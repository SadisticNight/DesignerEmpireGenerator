import capnp; ec=capnp.load("edificios.capnp")  # → Carga esquema Cap'n Proto de edificios
def le(a):  # → Función 'le': Lee binario y obtiene lista de edificios
    d=ec.Edificios.read(open(a,"rb"))  # Lee archivo binario y obtiene mensaje de edificios
    n=["suelo","residencia","taller_togas","herreria","decoracion","lecheria","depuradora","agua","refineria","policia","bombero","colegio","hospital"]  # → Nombres predefinidos
    # → Itera sobre la lista y muestra cada edificio formateado, usando compresión para efectos secundarios
    [print(f"{n[i] if i<len(n) else f'edificio_{i}'}:"+
           f"{{'color':{list(x.color)},'energia':{x.energia},'agua':{x.agua},'basura':{x.basura},'comida':{x.comida},"
           f"'empleos':{x.empleos},'residentes':{x.residentes},'tipo':'{x.tipo}','felicidad':{x.felicidad},"
           f"'ambiente':{x.ambiente},'tamanio':{list(x.tamanio)}}}") for i,x in enumerate(d.lista)]
if __name__=="__main__": le("edificios.bin")  # → Ejecuta la lectura y muestra resultados
