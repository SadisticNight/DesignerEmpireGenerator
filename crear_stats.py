import pickle

# Lista de edificios como se especificó en edificios.pkl (simulación del contenido)
nombres_edificios = [
    "suelo", "residencia", "taller_togas", "herreria", "decoracion", "lecheria", 
    "depuradora", "agua", "refineria", "policia", "bombero", "colegio", "hospital"
]

def generar_datos_stats():
    # Inicializando el diccionario de estadísticas
    stats = {
        "total_residentes": 0,
        "total_empleos_industria": 0,
        "total_empleos_comercio": 0,
        "total_empleos": 0,
        "porcentaje_industria": 0.0,
        "porcentaje_comercio": 0.0,
        "EC_proporcion_correcta": False,
        "energia_usada": 0,
        "energia_total": 0,
        "agua_usada": 0,
        "agua_total": 0,
        "comida_usada": 0,
        "comida_total": 0,
        "basura_usada": 0,
        "basura_total": 0,
        "ecologia_total": 0,
        "cantidad_edificios": {nombre: 0 for nombre in nombres_edificios}
    }
    return stats

# Generar datos de estadísticas iniciales
datos_stats = generar_datos_stats()

# Guardar los datos en un archivo pickle
with open('estadisticas.pkl', 'wb') as f:
    pickle.dump(datos_stats, f)

print("Archivo estadisticas.pkl generado y guardado correctamente.")
