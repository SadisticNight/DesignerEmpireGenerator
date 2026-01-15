# restricciones.py
from typing import Dict,Tuple,Any,Set,Iterable,Optional,List
from config import BOARD_SIZE
import edificios

S=int(BOARD_SIZE)

# --- CONFIGURACIÓN DE REGLAS ---

# 1. Edificios que requieren DECORACIÓN adyacente para funcionar
NECESITA_DECORACION={'agua'}

# 2. Edificios que requieren SUELO (Carretera) adyacente para funcionar
NECESITA_SUELO={
    'residencia',
    'taller_togas', 'herreria', 
    'hospital', 'colegio', 'policia', 'bombero',
    'refineria', 'lecheria', 'depuradora'
}

# 3. Capacidades de los proveedores de recursos
CAPACIDADES_PROVEEDOR={
    'refineria': {'energia':10000},
    'agua':      {'agua':40000},
    'lecheria':  {'comida':4000},
    'depuradora':{'basura':40000},
}
PROVEEDORES=set(CAPACIDADES_PROVEEDOR.keys())

# 4. Edificios de Servicio (Radio de efecto)
SERVICIOS={'policia','bombero','colegio','hospital'}

# --- VECTORES DE DIRECCIÓN ---
_LADOS=[(-1,0),(1,0),(0,-1),(0,1)]
_DIAGS=[(-1,-1),(-1,1),(1,-1),(1,1)]
_TODOS8=_LADOS+_DIAGS

# Cache de requisitos (se llena dinámicamente si se necesita)
REQ_ADJ_BY_BN={}

# --- FUNCIONES DE UTILIDAD BÁSICA ---

def nombre_en(board,pos):
    """Devuelve el nombre del edificio en una posición, manejando tuplas o strings."""
    v=board.get(pos)
    if v is None:return None
    # Si es tupla (nombre, id), devuelve nombre. Si es string, devuelve string.
    NAME_READERS={tuple:lambda t:t[0],str:lambda s:s}
    return NAME_READERS.get(type(v),lambda _:None)(v)

def requiere_suelo(bn):
    return bn.lower() in NECESITA_SUELO

def requiere_decoracion(bn):
    return bn.lower() in NECESITA_DECORACION

def obtener_bloque(x,y,w,h):
    """Devuelve el conjunto de coordenadas que ocupa un edificio de w*h."""
    xr=range(max(0,x),min(S,x+w))
    yr=range(max(0,y),min(S,y+h))
    return {(xi,yj) for xi in xr for yj in yr}

def vecinos_de_bloque(bloque,incluir_diagonales=True):
    """Devuelve las coordenadas vecinas a un conjunto de celdas (bloque)."""
    offsets=_TODOS8 if incluir_diagonales else _LADOS
    B=set(bloque)
    vecinos = set()
    for (x,y) in B:
        for (dx,dy) in offsets:
            nx, ny = x+dx, y+dy
            # Verificar limites y que no sea parte del propio edificio
            if 0<=nx<S and 0<=ny<S and (nx,ny) not in B:
                vecinos.add((nx,ny))
    return vecinos

def hay_tipo_adjacente(board,bloque,tipo_objetivo,incluir_diagonales=True):
    """Verifica si alguno de los vecinos del bloque es del tipo_objetivo."""
    vecs = vecinos_de_bloque(bloque,incluir_diagonales)
    for pos in vecs:
        bn = nombre_en(board, pos)
        if bn is not None and bn.lower() == tipo_objetivo:
            return True
    return False

# --- LÓGICA DE ESTADO (ACTIVO/INACTIVO) ---

def es_activo(board,nombre,bloque):
    """
    Determina si un edificio está activo basándose en sus requisitos de adyacencia.
    NO verifica recursos globales (eso lo hace el sistema de estadísticas).
    """
    bn=nombre.lower()
    
    # 1. Verificación de Suelo
    if requiere_suelo(bn):
        if not hay_tipo_adjacente(board, bloque, 'suelo', incluir_diagonales=True):
            return False
            
    # 2. Verificación de Decoración
    if requiere_decoracion(bn):
        if not hay_tipo_adjacente(board, bloque, 'decoracion', incluir_diagonales=True):
            return False
            
    return True

# --- REGLAS DE CONSTRUCCIÓN AUTOMÁTICA (PATRÓN 1001001) ---

def validar_distancia_suelo(board, x, y):
    """
    Impone la regla de densidad de carreteras:
    Un suelo NO puede tener otro suelo en un radio de 2 celdas (cuadrado 5x5).
    Excepción: Si la celda (x,y) ya tiene suelo (reemplazo), no cuenta.
    """
    # Rango de chequeo: x-2 a x+2
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0: continue # Ignorar la propia celda central
            
            nx, ny = x + i, y + j
            # Verificar limites del mapa
            if 0 <= nx < S and 0 <= ny < S:
                if (nx, ny) in board:
                    val = board[(nx, ny)]
                    # Extraer nombre limpiamente
                    bn = val[0] if isinstance(val, tuple) else str(val)
                    
                    if bn == 'suelo':
                        # ¡Violación encontrada! Hay un suelo demasiado cerca.
                        return False
    return True

# --- UTILIDADES PARA MODO MANUAL Y PROBABILIDADES ---

def capacidad_proveedor(nombre):
    return CAPACIDADES_PROVEEDOR.get(nombre.lower(),{}).copy()

def bloque_libre(board,bloque):
    return all(c not in board for c in bloque)

def score_prior(bn,lock_resources=(),servicio_coverage=None):
    """Calcula un multiplicador de probabilidad basado en cobertura de servicios."""
    b=bn.lower()
    k=1.0
    cov=float('nan')
    
    if servicio_coverage and b in SERVICIOS:
        cov=float(max(.0,min(1.,servicio_coverage.get(b,.0))))
        
    # Lógica de ajuste de probabilidad inversa a la cobertura
    # Si hay poca cobertura (<0.3), aumenta prob (1.3)
    # Si hay mucha cobertura (>0.85), baja prob (0.85)
    k *= 1.3 ** int(cov < .3)
    k *= 1.15 ** int(.3 <= cov and cov < .6)
    k *= .85 ** int(cov > .85)
    
    return float(max(.25,min(1.5,k)))

def reweight_probs(probs,action_names,lock_resources=(),servicio_coverage=None,provider_exists=None):
    """Recalcula probabilidades para la IA (usado en ml_policy/modo manual)."""
    names=list(action_names)
    n=len(names)
    if n==0:return []
    
    weights=[]
    for (p, name) in zip(probs, names):
        w = max(.0, float(p)) * score_prior(name, lock_resources, servicio_coverage)
        weights.append(w)
        
    # (Opcional) Filtrado por recursos bloqueados si fuera necesario
    # ...
    
    total=float(sum(weights))
    inv=1./total if total>.0 else .0
    
    # Normalizar
    return [w*inv if total>.0 else 1./n for w in weights]