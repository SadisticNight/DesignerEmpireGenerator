# restricciones.py
from typing import Dict,Tuple,Any,Set,Iterable,Optional,List
from config import BOARD_SIZE

S=int(BOARD_SIZE)

NECESITA_SUELO={'residencia','herreria','taller_togas','hospital','colegio','policia'}
NECESITA_DECORACION={'agua'}

CAPACIDADES_PROVEEDOR={
    'refineria': {'energia':10000},
    'agua':      {'agua':40000},
    'lecheria':  {'comida':4000},
    'depuradora':{'basura':40000},
}
PROVEEDORES=set(CAPACIDADES_PROVEEDOR.keys())

SERVICIOS={'policia','bombero','colegio','hospital'}

DEFICIT_A_PROVEEDOR={'energia':'refineria','agua':'agua','comida':'lecheria','basura':'depuradora'}
LOCK_TO_PROVIDER={'energia':'refineria','agua':'agua','comida':'lecheria','basura':'depuradora'}

_LADOS=[(-1,0),(1,0),(0,-1),(0,1)]
_DIAGS=[(-1,-1),(-1,1),(1,-1),(1,1)]
_TODOS8=_LADOS+_DIAGS

REQ_ADJ_BY_BN={}

def nombre_en(board,pos):
    v=board.get(pos)
    if v is None:return
    NAME_READERS={tuple:lambda t:t[0],str:lambda s:s}
    return NAME_READERS.get(type(v),lambda _:None)(v)

def requiere_suelo(bn):return bn.lower()in NECESITA_SUELO
def requiere_decoracion(bn):return bn.lower()in NECESITA_DECORACION

def obtener_bloque(x,y,w,h):
    xr=range(max(0,x),min(S,x+w)); yr=range(max(0,y),min(S,y+h))
    return {(xi,yj) for xi in xr for yj in yr}

def vecinos_de_bloque(bloque,incluir_diagonales=True):
    offsets=_TODOS8 if incluir_diagonales else _LADOS
    B=set(bloque)
    return {(nx,ny) for(x,y)in B for(dx,dy)in offsets
            for(nx,ny)in[(x+dx,y+dy)]
            if 0<=nx<S and 0<=ny<S and (nx,ny)not in B}

def hay_tipo_adjacente(board,bloque,tipo_objetivo,incluir_diagonales=True):
    return any((bn:=nombre_en(board,pos))is not None and bn.lower()==tipo_objetivo
               for pos in vecinos_de_bloque(bloque,incluir_diagonales))

def es_activo(board,nombre,bloque):
    # Activo SOLO por prerrequisitos (suelo/decoración). No depende de recursos.
    bn=nombre.lower()
    if not REQ_ADJ_BY_BN:
        for n in NECESITA_SUELO|NECESITA_DECORACION|SERVICIOS|PROVEEDORES|{'suelo','decoracion','demoler'}:
            REQ_ADJ_BY_BN[n]={'suelo'} if n in NECESITA_SUELO else {'decoracion'} if n in NECESITA_DECORACION else None
    req=REQ_ADJ_BY_BN.get(bn)
    return True if req is None else hay_tipo_adjacente(board,bloque,next(iter(req)),incluir_diagonales=True)

def capacidad_proveedor(nombre):return CAPACIDADES_PROVEEDOR.get(nombre.lower(),{}).copy()

def bloque_libre(board,bloque):return all(c not in board for c in bloque)

def score_prior(bn,lock_resources=(),servicio_coverage=None):
    b=bn.lower(); k=1.; cov=float('nan')
    if servicio_coverage and b in SERVICIOS:
        cov=float(max(.0,min(1.,servicio_coverage.get(b,.0))))
    k*=1.3**int(cov<.3); k*=1.15**int(.3<=cov and cov<.6); k*=.85**int(cov>.85)
    return float(max(.25,min(1.5,k)))

def reweight_probs(probs,action_names,lock_resources=(),servicio_coverage=None,provider_exists=None):
    names=list(action_names); n=len(names)
    if n==0:return []
    weights=[max(.0,float(p))*score_prior(name,lock_resources=lock_resources,servicio_coverage=servicio_coverage)
             for(p,name)in zip(probs,names)]
    if lock_resources and provider_exists is not None:
        locks={str(x).lower()for x in lock_resources}
        exists={str(x).lower()for x in provider_exists}
        for(i,name)in enumerate(names):
            nn=str(name).lower()
            for(r,p)in LOCK_TO_PROVIDER.items():
                if r in locks and nn==p and nn in exists:
                    weights[i]=.0
    total=float(sum(weights)); inv=1./total if total>.0 else .0
    return [w*inv if total>.0 else 1./n for w in weights]
