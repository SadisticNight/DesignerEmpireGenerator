# planificador.py
from __future__ import annotations
from typing import Dict,Tuple,Iterable,Optional,List,Set,Callable
import uuid,random,edificios,bd_celdas,areas,restricciones as R,scorer_jax
from estadisticas import StatsProcessor
from config import BOARD_SIZE,LOG_COLOCACIONES,LOG_FALLOS
Coord=Tuple[int,int]
Board=Dict[Coord,object]
S=int(BOARD_SIZE)
ED=edificios.edificios
SERVICIOS='policia','bombero','colegio','hospital'
UMBRAL_SOLAPE=.3
def _tam(bn):w,h=ED[bn].tamanio;return int(w),int(h)
def _bloque(x,y,w,h):return[]if x<0 or y<0 or x+w>S or y+h>S else[(x+i,y+j)for i in range(w)for j in range(h)]
def _libre(board,coords):return all(c not in board for c in coords)
def _log_colocado(bn,coords):
	if LOG_COLOCACIONES:
		try:lst=list(coords);print(f"Edificio {bn} construido en {lst[:3]}{"..."if len(lst)>3 else""}")
		except Exception:pass
def _log_fallo(msg):
	if LOG_FALLOS:
		try:print(msg)
		except Exception:pass
def _origenes(board):
	origins={};out={}
	for((x,y),val)in board.items():
		if isinstance(val,tuple):bn,raw_iid=val;iid=str(raw_iid);ox,oy=origins.get(iid,(x,y));origins[iid]=(min(ox,x),min(oy,y))if iid in origins else(x,y)
	for((x,y),val)in board.items():
		if isinstance(val,tuple):
			bn,raw_iid=val;iid=str(raw_iid)
			if origins.get(iid)==(x,y):out.setdefault(bn.lower(),[]).append((x,y))
		elif isinstance(val,str)and val:out.setdefault(val.lower(),[]).append((x,y))
	return out
def _union_cobertura(board,bn_srv):
	res=set()
	for(ox,oy)in _origenes(board).get(bn_srv.lower(),[]):
		for c in areas.Area.zona_cubierta_por_edificio(bn_srv,(ox,oy),S):res.add(c)
	return res
def _solape_relativo(board,bn_srv,pos):
	zona=areas.Area.zona_cubierta_por_edificio(bn_srv,pos,S);zlen=len(zona)
	if zlen==0:return 1.
	union=_union_cobertura(board,bn_srv);inter=sum(1 for c in zona if c in union);return inter/float(zlen)
_stats=lambda:StatsProcessor().process()
def _update_cells(board):bd_celdas.update_celdas_bin(board,lock_resources=set())
def _can_place_base(board,bn_low,x,y,w,h):coords=_bloque(x,y,w,h);return(False,'out_of_bounds')if not coords else(False,'occupied')if not _libre(board,coords)else(True,'ok')
def _can_place_servicio(board,bn_low,x,y,w,h):
	ok,reason=_can_place_base(board,bn_low,x,y,w,h)
	if not ok:return ok,reason
	sol=_solape_relativo(board,bn_low,(x,y));return(False,f"service_overlap:{sol:.3f}")if sol>UMBRAL_SOLAPE else(True,'ok')
_CAN_PLACE_DISPATCH={**{srv:_can_place_servicio for srv in SERVICIOS}}
def can_place(board,bn,x,y):w,h=_tam(bn);bn_low=bn.lower();handler=_CAN_PLACE_DISPATCH.get(bn_low,_can_place_base);return handler(board,bn_low,x,y,w,h)
def _place_simple(board,bn,x,y,w,h):board[x,y]=bn;_log_colocado(bn,[(x,y)]);return True
def _place_multi(board,bn,x,y,w,h):coords=_bloque(x,y,w,h);iid=str(uuid.uuid4());board.update({c:(bn,iid)for c in coords});_log_colocado(bn,coords);return True
def place(board,bn,x,y):
	ok,reason=can_place(board,bn,x,y)
	if not ok:_log_fallo(f"[rechazado] {bn} en ({x},{y}) → {reason}");return False
	w,h=_tam(bn);is_simple_1x1=bn in('suelo','decoracion')and(w==1 and h==1);return(_place_simple if is_simple_1x1 else _place_multi)(board,bn,x,y,w,h)
def _demoler_tuple(board,pos,v):
	bn,iid=v
	for(c,w)in list(board.items()):
		if w==(bn,iid):del board[c]
	if LOG_FALLOS:print(f"Demolido {bn} (iid={iid})")
	return True
def _demoler_str(board,pos,v):
	bn=str(v);del board[pos]
	if LOG_FALLOS:print(f"Demolido {bn} (1x1)")
	return True
_DEMOLER_DISPATCH={True:_demoler_tuple,False:_demoler_str}
def demoler(board,pos):v=board.get(pos);return False if v is None else _DEMOLER_DISPATCH[isinstance(v,tuple)](board,pos,v)
def _vecindad_5x5(cx,cy,w,h):
	vals=[]
	for dx in(-2,-1,0,1,2):
		for dy in(-2,-1,0,1,2):
			x,y=cx+dx,cy+dy
			if 0<=x<=S-w and 0<=y<=S-h:vals.append((x,y))
	return vals
def _gauss_candidatos(w,h,k=18):
	cx=cy=(S-1)//2;mk=[]
	for _ in range(k):rx=max(0,min(S-w,int(random.gauss(cx,S*.15))));ry=max(0,min(S-h,int(random.gauss(cy,S*.15))));mk.append((rx,ry))
	return mk
def _jitter_alrededor(puntos,w,h):
	for(x0,y0)in puntos[:6]:
		for dx in(-1,0,1):
			for dy in(-1,0,1):
				x,y=x0+dx,y0+dy
				if 0<=x<=S-w and 0<=y<=S-h:yield(x,y)
def _colocar_cumpliendo_reglas(board,bn,prefer_cerca=None):
	w,h=_tam(bn);candidatos=[]
	try:
		spot=scorer_jax.best_spot(board,bn,{k:_tam(k)for k in ED})
		if spot:candidatos.append(spot)
	except Exception:pass
	candidatos+=_vecindad_5x5(*prefer_cerca,w,h)if prefer_cerca else[];candidatos+=_gauss_candidatos(w,h,18)
	for(x,y)in candidatos:
		if place(board,bn,x,y):return True
	for(x,y)in _jitter_alrededor(candidatos,w,h):
		if place(board,bn,x,y):return True
	return False
def cobertura_union(board,bn_srv):return _union_cobertura(board,bn_srv)
def es_activo(board,bn,x,y):
	w,h=_tam(bn);bloque=_bloque(x,y,w,h)
	if not bloque:return False
	try:return bool(R.es_activo(board,bn,bloque))
	except Exception:return True
def plan_step(board):
	_update_cells(board);st=_stats();tot_usa={'energia':(st.energiaTotal,st.energiaUsada,'refineria'),'agua':(st.aguaTotal,st.aguaUsada,'agua'),'comida':(st.comidaTotal,st.comidaUsada,'lecheria'),'basura':(st.basuraTotal,st.basuraUsada,'depuradora')};deficit_val=lambda T,U:max(0,-(T+U));deficits=[(prov,deficit_val(T,U))for(T,U,prov)in tot_usa.values()];deficits=[(bn,g)for(bn,g)in deficits if g>0]
	if deficits:bn=max(deficits,key=lambda t:t[1])[0];return _colocar_cumpliendo_reglas(board,bn)
	contar=lambda b:sum(1 for v in board.values()if isinstance(v,tuple)and v[0]==b);menos=min(SERVICIOS,key=contar);return _colocar_cumpliendo_reglas(board,menos)
def plan_ciudad(board,seed=None):
	random.seed(seed)if seed is not None else None
	for _ in range(2000):
		if not plan_step(board):break
	return board