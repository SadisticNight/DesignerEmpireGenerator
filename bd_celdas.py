# bd_celdas.py
from typing import Any,Dict,List,Tuple,Set,Optional
import os,capnp,edificios
from config import BOARD_SIZE

celdas_schema=capnp.load('celdas.capnp')
SchemaMapa=celdas_schema.Mapa

NEED_ADJ_SUELO={'residencia','herreria','taller_togas','hospital','colegio','policia'}
NEED_ADJ_DECOR={'agua'}
LOCK_PROVIDERS={'energia':'refineria','agua':'agua','comida':'lecheria','basura':'depuradora'}

_NEIGH=tuple((dx,dy)for dx in(-1,0,1)for dy in(-1,0,1)if not(dx==0 and dy==0))
U64_MASK=(1<<64)-1
IID56_MASK=(1<<56)-1

_CELDST=os.getenv('DEG_CELDAS_PATH','celdas.bin')
_CELTMP=_CELDST+'.tmp'
_FLUSH_EVERY=1
_CALLS=0
_STRICT_ADJ=int(os.getenv('DEG_STRICT_ADJ','0'))!=0

_SNAPSHOT_BYTES=None
_SNAPSHOT_MSG=None

def get_snapshot_bytes():return _SNAPSHOT_BYTES
def get_snapshot_msg():return _SNAPSHOT_MSG

def force_flush():
	if _SNAPSHOT_BYTES is None:return
	_write_bytes_atomic(_SNAPSHOT_BYTES)

def _write_bytes_atomic(payload):
	os.makedirs(os.path.dirname(_CELDST)or'.',exist_ok=True)
	with open(_CELTMP,'wb')as f:f.write(payload)
	try:os.replace(_CELTMP,_CELDST)
	except Exception:
		try:os.rename(_CELTMP,_CELDST)
		except Exception:pass

def read_capnp_file(fn):
	with open(fn,'rb')as f:return SchemaMapa.read(f)

def write_capnp_file(fn,data):
	with open(fn,'wb')as f:f.write(data.to_bytes())

def _is_name(val,name):return isinstance(val,tuple)and val[0]==name or val==name
def _any_adjacent(board_get,x,y,required):return any(_is_name(board_get((x+dx,y+dy)),nm)for(dx,dy)in _NEIGH for nm in required)

def _block_has_required_adj(board_get,origin,size,required):
	x0,y0=origin;w,h=size
	for i in range(w):
		for j in range(h):
			if _any_adjacent(board_get,x0+i,y0+j,required):return True
	return False

def _to_u64_iid(raw_iid):
	try:
		if isinstance(raw_iid,int):return raw_iid&U64_MASK
		if isinstance(raw_iid,(bytes,bytearray)):return int.from_bytes(bytes(raw_iid[:8]).ljust(8,b'\x00'),'little',signed=False)&U64_MASK
		if isinstance(raw_iid,str):
			s=raw_iid.strip()
			if not s:return 0
			if'-'in s:
				import uuid
				try:return uuid.UUID(s).int&U64_MASK
				except Exception:pass
			try:return int(s,16)&U64_MASK
			except ValueError:return int(s,10)&U64_MASK
		return int(raw_iid)&U64_MASK
	except Exception:return hash(str(raw_iid))&U64_MASK

def update_celdas_bin(board,lock_resources=None):
	import areas,math
	global _CALLS,_SNAPSHOT_BYTES,_SNAPSHOT_MSG
	S=int(BOARD_SIZE);T=S*S
	msg=SchemaMapa.new_message();cel=msg.init('celdas',T)
	board_get=board.get
	eds=edificios.edificios;eds_get=eds.get
	z_srv=areas.Area.zona_cubierta_por_edificio
	z1=areas.Area.area_afectada_simple
	z2=areas.Area.area_afectada_2x2
	dat_cache={}
	REQ_ADJ_BY_BN={}
	for k in eds.keys():
		kl=k.lower()
		REQ_ADJ_BY_BN[kl]={'suelo'}if kl in NEED_ADJ_SUELO else{'decoracion'}if kl in NEED_ADJ_DECOR else None

	# origen (x,y) superior-izquierdo por iid
	origins={}
	for((x,y),b)in tuple(board.items()):
		if isinstance(b,tuple):
			_,raw_iid=b;iid=str(raw_iid);ox,oy=origins.get(iid,(x,y))
			if x<ox or y<oy:origins[iid]=min(x,ox),min(y,oy)
			else:origins.setdefault(iid,(x,y))

	# conjunto de “activos” SOLO por prerequisitos (no por recursos)
	if _STRICT_ADJ:
		active=set()
		for((x,y),b)in tuple(board.items()):
			if not isinstance(b,tuple):continue
			bn,raw_iid=b;iid=str(raw_iid);key=iid or(x,y);bo=eds_get(bn)
			if not bo:continue
			w,h=tuple(map(int,getattr(bo,'tamanio',(1,1))))
			ox,oy=origins.get(iid,(x,y))
			req=REQ_ADJ_BY_BN.get(str(bn).lower())
			should_add=req is None or _block_has_required_adj(board_get,(ox,oy),(w,h),req)
			if should_add:active.add(key)
	else:
		active={str(b[1])or(x,y)for((x,y),b)in tuple(board.items())if isinstance(b,tuple)}

	# ÁREAS DE SERVICIO: SIEMPRE activas (solo desde origen)
	svc_idx={'policia':0,'bombero':1,'hospital':2,'colegio':3}
	svc_map={}
	for((x,y),b)in tuple(board.items()):
		if not isinstance(b,tuple):continue
		bn,raw_iid=b;k=svc_idx.get(str(bn).lower(),-1)
		if k<0:continue
		bo=eds_get(bn)
		if not bo:continue
		dat=dat_cache.get(bn)or bo.to_dict;dat_cache.setdefault(bn,dat)
		w,h=map(int,dat['tamanio'])
		iid_str=str(raw_iid);ox,oy=origins.get(iid_str,(x,y))
		is_origin=w==1 and h==1 or(w>1 and h>1)and(x,y)==(ox,oy)
		if not is_origin:continue
		for(cx,cy)in z_srv(bn,(ox,oy),S):
			v=svc_map.get((cx,cy))
			if v is None:v=[0,0,0,0];svc_map[cx,cy]=v
			v[k]+=1

	def _zone_points(bn,ox,oy,w,h):
		if w>1 and h>1:return z2(bn,(ox,oy),S),(ox+.5,oy+.5)
		else:return z1(bn,(ox,oy),S),(ox,oy)

	# Mapas acumulados de ambiente (siempre) y felicidad (solo activos)
	amb_map={};fel_map={}
	for((x,y),val)in tuple(board.items()):
		if isinstance(val,tuple):
			bn,raw_iid=val;iid=str(raw_iid);ox,oy=origins.get(iid,(x,y))
			if(x,y)!=(ox,oy):continue
		else:
			bn=str(val);ox,oy=x,y
		bo=eds_get(bn)
		if not bo:continue
		dat=dat_cache.get(bn)or bo.to_dict;dat_cache.setdefault(bn,dat)
		w,h=map(int,dat['tamanio'])
		pts,(cx,cy)=_zone_points(bn,ox,oy,w,h)
		R2=.0
		for(px,py)in pts:
			dx=px-cx;dy=py-cy;d2=dx*dx+dy*dy
			if d2>R2:R2=d2
		invR=1./(R2**.5 if R2>.0 else 1.)
		base_fel=int(dat['felicidad']);base_amb=int(dat['ambiente'])

		# Ambiente SIEMPRE suma (activos o no)
		if base_amb:
			for(px,py)in pts:
				dx=px-cx;dy=py-cy
				f=.5+.5*(1-min((dx*dx+dy*dy)**.5*invR,1.))
				amb_map[px,py]=amb_map.get((px,py),0)+round(base_amb*f)

		# Felicidad solo si el edificio está “activo” por prerequisitos (no por recursos)
		if base_fel:
			if isinstance(val,tuple):
				key=str(val[1])or(x,y)
				if key not in active:continue
			for(px,py)in pts:
				dx=px-cx;dy=py-cy
				f=.5+.5*(1-min((dx*dx+dy*dy)**.5*invR,1.))
				fel_map[px,py]=fel_map.get((px,py),0)+round(base_fel*f)

	eds_get_local=eds_get;origins_local=origins;active_local=active

	def _write_none(c):c.hash=0;c.edificio='';c.tipo='vacio'
	def _write_str(c,bn):c.edificio=bn;c.tipo='vacio';c.hash=0

	def _write_tuple(x,y,c,val):
		bn,raw_iid=val;bo=eds_get_local(bn)
		if not bo:c.tipo='vacio';c.hash=0;c.edificio='';return
		c.edificio=str(bn);c.tipo=str(bo.tipo.value)
		dat=dat_cache.get(bn)or bo.to_dict;dat_cache.setdefault(bn,dat)
		w,h=map(int,dat['tamanio'])
		iid_str=str(raw_iid);ox,oy=origins_local.get(iid_str,(x,y))
		is_origin=w==1 and h==1 or(w>1 and h>1)and(x,y)==(ox,oy)
		key=iid_str or(x,y);is_active=key in active_local
		if is_active:
			iid_u56=_to_u64_iid(raw_iid)&IID56_MASK
			tile_idx=x-ox+(y-oy)*max(w,1);tile_idx&=255
			c.hash=(iid_u56<<8|tile_idx)&U64_MASK
		else:
			c.hash=0
		if is_active and is_origin:
			at=c.atributos
			at.energia+=dat['energia'];at.agua+=dat['agua']
			at.basura+=dat['basura'];at.comida+=dat['comida']
			at.empleos+=dat['empleos'];at.residentes+=dat['residentes']

	CELL_WRITERS={
		type(None):lambda x,y,c,v=None:_write_none(c),
		str:lambda x,y,c,v:_write_str(c,v),
		tuple:_write_tuple
	}

	for i in range(T):
		x,y=divmod(i,S);c=cel[i];c.x=x;c.y=y
		at=c.atributos
		at.energia=at.agua=at.basura=at.comida=0
		at.empleos=at.residentes=0
		at.felicidad=at.ambiente=0
		val=board_get((x,y))
		CELL_WRITERS.get(type(val),CELL_WRITERS[type(None)])(x,y,c,val)
		fa=fel_map.get((x,y),0);am=amb_map.get((x,y),0)
		at.felicidad+=fa;at.ambiente+=am
		sv=c.servicios
		sv.seguridad,sv.incendio,sv.salud,sv.educacion=tuple(svc_map.get((x,y))or(0,0,0,0))

	payload=msg.to_bytes()
	_SNAPSHOT_MSG=msg;_SNAPSHOT_BYTES=payload
	_CALLS+=1
	if _CALLS%_FLUSH_EVERY==0:_write_bytes_atomic(payload)
	force_flush()
	return msg
