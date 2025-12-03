# atributos.py
from types import MappingProxyType

class Atributo:
    __slots__=('color','energia','agua','basura','comida','empleos','residentes','tipo','felicidad','ambiente','tamanio','_d','_p')

    def __init__(self,color,energia,agua,basura,comida,empleos,residentes,tipo,felicidad,ambiente,tamanio):
        object.__setattr__(self,'_d',{})  # cache primero
        object.__setattr__(self,'_p',MappingProxyType(self._d))
        self.color=color; self.energia=energia; self.agua=agua; self.basura=basura; self.comida=comida
        self.empleos=empleos; self.residentes=residentes; self.tipo=tipo; self.felicidad=felicidad; self.ambiente=ambiente; self.tamanio=tamanio

    def __setattr__(self,k,v):
        object.__setattr__(self,k,v)
        if k in ('color','energia','agua','basura','comida','empleos','residentes','tipo','felicidad','ambiente','tamanio'):
            try: self._d[k]=v
            except AttributeError: pass  # durante bootstrap

    @property
    def to_dict(self):  # O(1) sin copias
        return self._p
