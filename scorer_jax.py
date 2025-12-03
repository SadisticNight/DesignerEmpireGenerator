# scorer_jax.py
import jax.numpy as jnp
from jax import jit,lax
from functools import partial
from typing import Dict,Tuple,Optional,Any
from contextlib import nullcontext
from config import BOARD_SIZE
from restricciones import obtener_bloque,es_activo
S=int(BOARD_SIZE)
INF_NEG=jnp.array(-1e30,dtype=jnp.float32)
def make_occupancy():return jnp.zeros((S,S),dtype=jnp.int32)
def _apply_rect(occ,x,y,w,h,val):return occ.at[x:x+w,y:y+h].set(jnp.int32(val))
def occ_mark_place(occ,x,y,w,h):return _apply_rect(occ,x,y,w,h,1)
def occ_mark_clear(occ,x,y,w,h):return _apply_rect(occ,x,y,w,h,0)
def _build_occupancy(board):
	coords=list(board.keys())
	def _empty():return jnp.zeros((S,S),dtype=jnp.int32)
	def _fill():xs,ys=zip(*coords);xs=jnp.array(xs,dtype=jnp.int32);ys=jnp.array(ys,dtype=jnp.int32);return jnp.zeros((S,S),dtype=jnp.int32).at[xs,ys].set(1)
	return{False:_empty,True:_fill}[bool(coords)]()
@partial(jit,static_argnums=(1,2,3))
def _rank_candidates(occ,w,h,topk):sat=jnp.pad(jnp.cumsum(jnp.cumsum(occ,0),1),((1,0),(1,0)));nx=S-w+1;ny=S-h+1;X=jnp.arange(nx)[:,None];Y=jnp.arange(ny)[None,:];blk=sat[X+w,Y+h]-sat[X,Y+h]-sat[X+w,Y]+sat[X,Y];valid=blk==0;cx=(S-1)*.5;cy=(S-1)*.5;Xc=(X+w*.5).astype(jnp.float32);Yc=(Y+h*.5).astype(jnp.float32);sc=-((Xc-cx)**2+(Yc-cy)**2);msc=jnp.where(valid,sc,INF_NEG).reshape(-1);size=nx*ny;k=min(int(topk),int(size));vals,idxs=lax.top_k(msc,k);idxs=idxs.astype(jnp.int32);ix=(idxs//ny).astype(jnp.int32);iy=(idxs%ny).astype(jnp.int32);cnt_valid=jnp.sum(valid).astype(jnp.int32);return ix,iy,cnt_valid,jnp.int32(k)
def best_spot(board,nombre,tamanios,topk=512,occ=None):
	w,h=map(int,tamanios.get(nombre,(1,1)))
	def _invalid():0
	def _search():occ_arr=occ if occ is not None else _build_occupancy(board);xs,ys,cnt,k=_rank_candidates(occ_arr,w,h,int(topk));n=int(min(int(k),int(cnt)));return next(((x,y)for i in range(n)for(x,y)in[(int(xs[i]),int(ys[i]))]if es_activo(board,nombre,obtener_bloque(x,y,w,h))),None)
	return{True:_search,False:_invalid}[1<=w<=S and 1<=h<=S]()
def apply_best_spot(board,nombre,tamanios,lock=None,occ=None):
	spot=best_spot(board,nombre,tamanios,topk=512,occ=occ)
	def _no_spot():return False,None
	def _try_apply():
		x,y=spot;w,h=map(int,tamanios.get(nombre,(1,1)));coords={(x+i,y+j)for i in range(w)for j in range(h)};ctx=lock if lock else nullcontext()
		with ctx:
			ok=coords.isdisjoint(board.keys())
			def _commit():iid=str(id((nombre,x,y)));board.update({(x+i,y+j):(nombre,iid)for i in range(w)for j in range(h)});return True,(x,y)
			def _fail():return False,None
			return{True:_commit,False:_fail}[ok]()
	return{True:_try_apply,False:_no_spot}[spot is not None]()