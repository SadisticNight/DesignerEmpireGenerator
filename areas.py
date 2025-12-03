# areas.py
import math,jax.numpy as jnp
from jax import jit
from functools import partial
from bd_celdas import write_capnp_file
RADIOS_AREA={k:4 for k in('residencia','taller_togas','herreria','lecheria','refineria','policia','bombero','colegio','hospital','decoracion')}
RADIOS_AREA_2X2={'depuradora':4,'agua':4}
RADIOS_SERVICIOS={'policia':13,'bombero':13,'colegio':13,'hospital':11}
RMAX=13
W=2*RMAX+1
W_SQUARED=W*W
OFFS=jnp.arange(-RMAX,RMAX+1,dtype=jnp.int32)
OFFX2D,OFFY2D=jnp.meshgrid(OFFS,OFFS,indexing='ij')
@partial(jit,static_argnums=(3,))
def _disk_int_window(x_c,y_c,r,N):x_c=jnp.int32(x_c);y_c=jnp.int32(y_c);r=jnp.int32(r);N=jnp.int32(N);X=jnp.clip(x_c+OFFX2D,0,N-1);Y=jnp.clip(y_c+OFFY2D,0,N-1);dx=X-x_c;dy=Y-y_c;r2=r*r;m=dx*dx+dy*dy<=r2;n=jnp.sum(m,dtype=jnp.int32);ii,jj=jnp.nonzero(m,size=W_SQUARED);return X[ii,jj],Y[ii,jj],n
@partial(jit,static_argnums=(3,))
def _disk_half_window(ax,ay,r,N):ax=jnp.float32(ax);ay=jnp.float32(ay);r=jnp.float32(r);N=jnp.int32(N);bx=jnp.floor(ax).astype(jnp.int32);by=jnp.floor(ay).astype(jnp.int32);X=jnp.clip(bx+OFFX2D,0,N-1);Y=jnp.clip(by+OFFY2D,0,N-1);Xf=X.astype(jnp.float32);Yf=Y.astype(jnp.float32);r2=r*r;m=(Xf-ax)*(Xf-ax)+(Yf-ay)*(Yf-ay)<=r2;n=jnp.sum(m,dtype=jnp.int32);ii,jj=jnp.nonzero(m,size=W_SQUARED);return X[ii,jj],Y[ii,jj],n
class Area:
	__slots__='area_afectada','area_cubierta','cords_edificio','max_radio_afectado','max_radio_cubierto','x_centro','y_centro','max_efecto'
	def __init__(self):self.area_afectada=self.area_cubierta=self.cords_edificio=set();self.max_radio_afectado=self.max_radio_cubierto=self.x_centro=self.y_centro=0;self.max_efecto=100
	@staticmethod
	def calcular_area(x_c,y_c,r,N):xs_pad,ys_pad,n=_disk_int_window(int(x_c),int(y_c),int(r),int(N));n=int(n);return set(zip(xs_pad[:n].tolist(),ys_pad[:n].tolist()))
	@staticmethod
	def area_afectada_simple(ed,pos,N=200):r=RADIOS_AREA.get(ed.lower(),0);A=Area();A.x_centro,A.y_centro=pos;A.cords_edificio.add(tuple(pos));A.max_radio_afectado=r;A.area_afectada=Area.calcular_area(A.x_centro,A.y_centro,r,N);return list(A.area_afectada)
	@staticmethod
	def area_afectada_2x2(ed,pos,N=200):r=RADIOS_AREA_2X2.get(ed.lower(),0);A=Area();x,y=pos;ax,ay=x+.5,y+.5;A.x_centro,A.y_centro=ax,ay;A.cords_edificio.add(tuple(pos));A.max_radio_afectado=r;xs_pad,ys_pad,n=_disk_half_window(ax,ay,float(r),int(N));n=int(n);A.area_afectada=set(zip(xs_pad[:n].tolist(),ys_pad[:n].tolist()));return list(A.area_afectada)
	@staticmethod
	def zona_cubierta_por_edificio(ed,pos,N=200):r=RADIOS_SERVICIOS.get(ed.lower(),0);A=Area();A.x_centro,A.y_centro=pos;A.cords_edificio.add(tuple(pos));A.max_radio_cubierto=r;A.area_cubierta=Area.calcular_area(A.x_centro,A.y_centro,r,N);return list(A.area_cubierta)
	@staticmethod
	def actualizar_celdas(ed,dat,msg,ctr,area):
		fx,fy=ctr;R2=max((x-fx)**2+(y-fy)**2 for(x,y)in area)or 1;invR=1/math.sqrt(R2)
		for(x,y)in area:d=math.sqrt((x-fx)**2+(y-fy)**2)*invR;f=.5+.5*(1-min(d,1.));c=msg.get_celda(x,y);c.felicidad+=round(dat['felicidad']*f);c.ambiente+=round(dat['ambiente']*f)
		return msg
	@staticmethod
	def actualizar_servicios(ed,msg,orig):
		map_srv={'policia':lambda c:setattr(c.servicios,'seguridad',c.servicios.seguridad+1),'bombero':lambda c:setattr(c.servicios,'incendio',c.servicios.incendio+1),'colegio':lambda c:setattr(c.servicios,'educacion',c.servicios.educacion+1),'hospital':lambda c:setattr(c.servicios,'salud',c.servicios.salud+1)}
		for(x,y)in Area.zona_cubierta_por_edificio(ed,orig):map_srv.get(ed.lower(),lambda c:None)(msg.get_celda(x,y))
		return msg
def actualizar_area_y_guardar(ed,dat,msg,ctr,area,board,es_2x2=False,orig=None,filename='celdas.bin'):
	for c in area:board.setdefault(c,('',None))
	center=orig if es_2x2 else ctr;msg=Area.actualizar_celdas(ed,dat,msg,center,area);msg=Area.actualizar_servicios(ed,msg,center);write_capnp_file(filename,msg);return msg