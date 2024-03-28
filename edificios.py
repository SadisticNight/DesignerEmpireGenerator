from enum import Enum
from atributos import Atributo
import pickle

class TipoEdificio(Enum):
    COMERCIO = "comercio"
    INDUSTRIA = "industria"
    RESIDENCIA = "residencia"
    DECORACION = "decoracion"
    SUELO = "suelo"

# Carga los edificios desde un archivo .pkl
with open('edificios.pkl', 'rb') as archivo:
    datos = pickle.load(archivo) 

edificios = {nombre: Atributo(color=atributos["color"], energia=atributos["energia"], agua=atributos["agua"], basura=atributos["basura"], comida=atributos["comida"], empleos=atributos["empleos"], residentes=atributos["residentes"], tipo=TipoEdificio[atributos["tipo"].upper()], felicidad=atributos["felicidad"], ambiente=atributos["ambiente"], tamanio=atributos["tamanio"]) for nombre, atributos in datos.items()}
