# ocupacion.py
from __future__ import annotations
import threading
from typing import Dict,Tuple,Optional
class OccupancyBuffer:
	__slots__='S','mapa_ocupacion_bits','edificio_mascaras_por_id','bloqueo','_rowmask_cache'
	def __init__(self,S):self.S=S;self.mapa_ocupacion_bits=0;self.edificio_mascaras_por_id={};self.bloqueo=threading.Lock();self._rowmask_cache={}
	def _row_mask(self,x,ancho):
		k=x,ancho;m=self._rowmask_cache.get(k)
		if m is None:
			if ancho<=0:return 0
			m=(1<<ancho)-1<<x;self._rowmask_cache[k]=m
		return m
	def _rect_mask(self,x,y,ancho,alto):
		if ancho<=0 or alto<=0:return 0
		r=self._row_mask(x,ancho);m=r;step=self.S
		for _ in range(1,alto):m|=m<<step
		return m<<y*step
	@staticmethod
	def _dentro(S,x,y,ancho,alto):return x>=0 and y>=0 and x+ancho<=S and y+alto<=S
	def puede_colocar(self,x,y,ancho,alto):
		if not self._dentro(self.S,x,y,ancho,alto):return False
		m=self._rect_mask(x,y,ancho,alto);return self.mapa_ocupacion_bits&m==0
	def colocar(self,identificador,x,y,ancho,alto):
		with self.bloqueo:
			if not self._dentro(self.S,x,y,ancho,alto):return False
			m=self._rect_mask(x,y,ancho,alto)
			if self.mapa_ocupacion_bits&m:return False
			self.mapa_ocupacion_bits|=m;self.edificio_mascaras_por_id[identificador]=m;return True
	def demoler(self,identificador):
		with self.bloqueo:
			m=self.edificio_mascaras_por_id.pop(identificador,None)
			if m is None:return False
			self.mapa_ocupacion_bits&=~m;return True
	def puede_colocar_1x1(self,x,y):return self.puede_colocar(x,y,1,1)
	def puede_colocar_2x2(self,x,y):return self.puede_colocar(x,y,2,2)
	def colocar_1x1(self,i,x,y):return self.colocar(i,x,y,1,1)
	def colocar_2x2(self,i,x,y):return self.colocar(i,x,y,2,2)
	def contar_celdas_ocupadas(self):return self.mapa_ocupacion_bits.bit_count()
	def celda_libre(self,x,y):return self.puede_colocar(x,y,1,1)
class MapaDeTipos:
	__slots__='S','buffer','bloqueo'
	def __init__(self,S):self.S=S;self.buffer=bytearray(S*S);self.bloqueo=threading.Lock()
	def set_rect(self,x,y,ancho,alto,valor):
		if ancho<=0 or alto<=0:return
		S=self.S;fila=bytes([valor])*ancho;base=y*S+x
		with self.bloqueo:
			for r in range(alto):i=base+r*S;self.buffer[i:i+ancho]=fila
	def get(self,x,y):return self.buffer[y*self.S+x]