# generar_edificios.py
import capnp
ec = capnp.load("edificios.capnp")  # Carga del esquema de edificios

def generar_edificios(a):
    # Genera un mensaje nuevo y carga la lista predefinida del esquema
    m = ec.Edificios.new_message()
    lst = ec.edificiosPredefinidos.lista
    m.init("lista", len(lst))  # Inicializa la lista con el tamaño adecuado
    for i in range(len(lst)):
        s = lst[i]  # Fuente (datos predefinidos)
        c = m.lista[i]  # Destino (mensaje a escribir)
        # Asigna todos los campos, incluyendo el campo 'nombre'
        c.nombre = s.nombre      # <<-- Asignación agregada para el nombre
        c.color, c.energia, c.agua, c.basura, c.comida, c.empleos, c.residentes, c.tipo, c.felicidad, c.ambiente, c.tamanio = \
            s.color, s.energia, s.agua, s.basura, s.comida, s.empleos, s.residentes, s.tipo, s.felicidad, s.ambiente, s.tamanio
    with open(a, "wb") as f:
        m.write(f)

if __name__=="__main__":
    generar_edificios("edificios.bin")
    print("Datos predefinidos guardados en 'edificios.bin'")
