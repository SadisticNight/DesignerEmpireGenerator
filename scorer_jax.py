# scorer_jax.py
import jax.numpy as jnp
from jax import jit,lax
from functools import partial
from contextlib import nullcontext
from config import BOARD_SIZE
from restricciones import obtener_bloque,es_activo

S=int(BOARD_SIZE)
INF_NEG=jnp.array(-1e30,dtype=jnp.float32)

def _mk_kern(r,v,linear=True):
    w=2*r+1;Y,X=jnp.ogrid[:w,:w];d=jnp.sqrt((X-r)**2+(Y-r)**2);m=d<=r
    eff=(1.-(d/r))*v if linear else jnp.full((w,w),v)
    return jnp.where(m,eff,0.).astype(jnp.float32)

KERS={'policia':_mk_kern(13,10),'bombero':_mk_kern(13,10),'decoracion':_mk_kern(4,5),'residencia':_mk_kern(4,-2),'colegio':_mk_kern(13,8),'hospital':_mk_kern(11,10)}

def make_occupancy():return jnp.zeros((S,S),dtype=jnp.int32)
def make_maps():return{'seguridad':jnp.zeros((S,S),dtype=jnp.float32),'felicidad':jnp.zeros((S,S),dtype=jnp.float32),'ambiente':jnp.zeros((S,S),dtype=jnp.float32)}

# --- CORRECCION CRITICA: SIN JIT AQUI PARA PERMITIR RECORTES EN BORDES ---
def apply_kernel_diff(maps,k_name,x,y,sign):
    if k_name not in KERS:return maps
    k=KERS[k_name];kh,kw=k.shape;r=kh//2
    # Convertimos a int de Python puro para evitar trazas de JAX en los indices
    xs,ys=int(x-r),int(y-r);xe,ye=int(xs+kw),int(ys+kh)
    
    # Calculos de recorte usando max/min de Python standard
    kxs,kys=max(0,-xs),max(0,-ys)
    kxe,kye=kw-max(0,xe-S),kh-max(0,ye-S)
    mxs,mys=max(0,xs),max(0,ys)
    mxe,mye=min(S,xe),min(S,ye)
    
    # Si el edificio esta totalmente fuera (raro), salir
    if kxe<=kxs or kye<=kys: return maps

    # Slicing dinámico (esto es lo que fallaba con JIT)
    kv=k[kxs:kxe,kys:kye]*sign
    tgt='seguridad' if k_name in('policia','bombero')else 'ambiente' if k_name=='decoracion'else 'felicidad'
    
    # Operación in-place de JAX (permitida en modo eager)
    maps[tgt]=maps[tgt].at[mxs:mxe,mys:mye].add(kv)
    return maps

def _apply_rect(occ,x,y,w,h,val):return occ.at[x:x+w,y:y+h].set(jnp.int32(val))
def occ_mark_place(occ,x,y,w,h):return _apply_rect(occ,x,y,w,h,1)
def occ_mark_clear(occ,x,y,w,h):return _apply_rect(occ,x,y,w,h,0)

def _build_occupancy(board):
    c=list(board.keys());e=lambda:jnp.zeros((S,S),dtype=jnp.int32)
    f=lambda:jnp.zeros((S,S),dtype=jnp.int32).at[jnp.array([x for x,_ in c]),jnp.array([y for _,y in c])].set(1)
    return{False:e,True:f}[bool(c)]()

# MANTENEMOS JIT AQUI PORQUE ES EL CEREBRO PESADO (MATRICES FIJAS)
@partial(jit,static_argnums=(1,2,3))
def _rank_candidates(occ,w,h,topk):
    sat=jnp.pad(jnp.cumsum(jnp.cumsum(occ,0),1),((1,0),(1,0)))
    nx,ny=S-w+1,S-h+1
    # Generacion de grilla estatica
    X=jnp.arange(nx, dtype=jnp.int32)[:,None]
    Y=jnp.arange(ny, dtype=jnp.int32)[None,:]
    
    blk=sat[X+w,Y+h]-sat[X,Y+h]-sat[X+w,Y]+sat[X,Y];valid=blk==0
    
    cx,cy=jnp.float32((S-1)*.5),jnp.float32((S-1)*.5)
    Xc=(X+w*.5).astype(jnp.float32);Yc=(Y+h*.5).astype(jnp.float32)
    
    sc=-((Xc-cx)**2+(Yc-cy)**2)
    msc=jnp.where(valid,sc,INF_NEG).reshape(-1)
    
    k=min(int(topk),int(nx*ny));vals,idxs=lax.top_k(msc,k)
    ix,iy=(idxs//ny).astype(jnp.int32),(idxs%ny).astype(jnp.int32)
    return ix,iy,jnp.sum(valid).astype(jnp.int32),jnp.int32(k)

def best_spot(board,nombre,tamanios,topk=512,occ=None):
    w,h=map(int,tamanios.get(nombre,(1,1)))
    if not (1<=w<=S and 1<=h<=S): return None
    
    o=occ if occ is not None else _build_occupancy(board)
    xs,ys,cnt,k=_rank_candidates(o,w,h,int(topk))
    n=int(min(k,cnt))
    
    # Convertimos a listas de Python para iterar (sacamos de JAX)
    xs_py=xs[:n].tolist(); ys_py=ys[:n].tolist()
    
    for i in range(n):
        x,y=xs_py[i],ys_py[i]
        if es_activo(board,nombre,obtener_bloque(x,y,w,h)):
            return (x,y)
    return None

def apply_best_spot(board,nombre,tamanios,lock=None,occ=None):
    spot=best_spot(board,nombre,tamanios,topk=512,occ=occ)
    def _fail():return False,None
    def _ok():
        x,y=spot;w,h=map(int,tamanios.get(nombre,(1,1)))
        coords={(x+i,y+j)for i in range(w)for j in range(h)}
        with(lock if lock else nullcontext()):
            if not coords.isdisjoint(board.keys()):return False,None
            iid=str(id((nombre,x,y)));board.update({c:(nombre,iid)for c in coords});return True,(x,y)
    return{True:_ok,False:_fail}[spot is not None]()