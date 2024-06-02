import numpy as np
import pickle

class Stats:
    __slots__ = ['archivo_celdas', 'archivo_estadisticas', 'proveedores', 'recursos_por_edificio', 'hashes_contados']

    def __init__(self, archivo_celdas='celdas.pkl', archivo_estadisticas='estadisticas.pkl'):
        self.archivo_celdas = archivo_celdas
        self.archivo_estadisticas = archivo_estadisticas
        self.proveedores = {
            'energia': ['refineria'],
            'agua': ['agua'],
            'comida': ['lecheria'],
            'basura': ['depuradora']
        }
        self.recursos_por_edificio = {
            'refineria': 10000,
            'agua': 40000,
            'lecheria': 4000,
            'depuradora': 40000
        }
        self.hashes_contados = set()

    @staticmethod
    def cargar_datos(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def guardar_datos(filename, data):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def procesar_estadisticas(self):
        # Cargar datos existentes
        estadisticas = self.cargar_datos(self.archivo_estadisticas)
        celdas = self.cargar_datos(self.archivo_celdas)['celdas']

        # Convertir listas a numpy arrays para operaciones vectorizadas
        celdas_arr = np.array(celdas)

        # Procesar cada celda
        for celda in celdas_arr:
            atributos = celda['atributos']
            edificio = celda['edificio']
            hash_value = celda['hash']

            if hash_value not in self.hashes_contados:
                estadisticas['total_residentes'] += atributos['residentes']
                estadisticas['total_empleos'] += atributos['empleos']
                estadisticas['desequilibrio_laboral'] = estadisticas['total_empleos'] - estadisticas['total_residentes']

                match edificio:
                    case 'industria':
                        estadisticas['total_empleos_industria'] += atributos['empleos']
                    case 'comercio':
                        estadisticas['total_empleos_comercio'] += atributos['empleos']
                    case _:
                        pass
                
                # Calcular proporción Employment Composition
                if estadisticas['total_empleos'] > 0:
                    total_empleos = estadisticas['total_empleos']
                    estadisticas['porcentaje_industria'] = (estadisticas['total_empleos_industria'] / total_empleos * 100)
                    estadisticas['porcentaje_comercio'] = (estadisticas['total_empleos_comercio'] / total_empleos * 100)
                    ratio_industria = estadisticas['total_empleos_industria'] / total_empleos
                    ratio_comercio = estadisticas['total_empleos_comercio'] / total_empleos
                    estadisticas['EC'] = (ratio_industria == 1/3 and ratio_comercio == 2/3 and -100 <= estadisticas['desequilibrio_laboral'] <= 100)

                # Acumular uso y total de recursos
                for recurso, edificios in self.proveedores.items():
                    estadisticas[f'{recurso}_usada'] += atributos[recurso]
                    if edificio in edificios:
                        estadisticas[f'{recurso}_total'] += self.recursos_por_edificio[edificio]

                # Acumular ecología total
                estadisticas['ecologia_total'] += atributos['ambiente']

                # Contar edificios
                if edificio in estadisticas['cantidad_edificios']:
                    estadisticas['cantidad_edificios'][edificio] += 1
                else:
                    estadisticas['cantidad_edificios'][edificio] = 1

                self.hashes_contados.add(hash_value)

        # Guardar estadísticas actualizadas
        self.guardar_datos(self.archivo_estadisticas, estadisticas)
