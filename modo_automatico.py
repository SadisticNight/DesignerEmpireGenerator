# modo_automatico.py
from __future__ import annotations
import os,time,threading,jax,math
import jax.numpy as jnp
import numpy as np 
import random
from config import BOARD_SIZE
import edificios
from ml_policy import Policy
import scorer_jax as scorer
import restricciones as R
from mapa_calor import AnalistaTerreno

S=int(BOARD_SIZE)

ACTION_SET=('residencia','taller_togas','herreria','refineria','lecheria','agua',
            'depuradora','decoracion','policia','bombero','colegio','hospital',
            'suelo','demoler')
IDX={a:i for i,a in enumerate(ACTION_SET)}
NA=len(ACTION_SET)
BN_TO_ID={k:i+1 for i,k in enumerate(ACTION_SET) if k!='demoler'}; BN_TO_ID['suelo']=0

STATS_CACHE={k:edificios.edificios[k] for k in edificios.edificios if k in ACTION_SET and k!='demoler' and k!='suelo'}
CHUNK_SIZE=20
_TODOS8 = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]

def _obs_fast(maps,counts,score=0.0):
    s=jnp.sum; m=maps
    v=[float(s(m['seguridad']))/1000., float(s(m['felicidad']))/1000.]
    c=[float(counts.get(k,0))/50. for k in ACTION_SET]
    return jnp.array(v+c+[score/100.],dtype=jnp.float32)[:3+NA]

def _rew_fast(maps):
    s=jnp.sum; m=maps
    return float(s(m['felicidad'])*1.0+s(m['seguridad'])*0.5+s(m['ambiente'])*0.2)

def _get_chunk_rect(idx):
    chunks_per_row=S//CHUNK_SIZE
    row=idx//chunks_per_row; col=idx%chunks_per_row
    x0,y0=col*CHUNK_SIZE,row*CHUNK_SIZE
    return x0,y0,x0+CHUNK_SIZE,y0+CHUNK_SIZE

def _tam(bn): 
    if bn in edificios.edificios:
        try:
            tamanio_list = list(edificios.edificios[bn].tamanio)
            return tuple(map(int, tamanio_list))
        except Exception:
            return (1, 1)
    return (1, 1)


class ModoAutomatico:
    __slots__=('board','running','thread','lock','agent','occ','jax_maps','steps',
               'rng_key','chunk_idx','cnt','active_mask','blacklist',
               'stuck_steps','analista_terreno', 'heatmap_cache', 'objetivo_actual',
               'foco_construccion', 'cache_suelos', 'cache_decos') 
    
    def __init__(self,board,lock=None):
        self.board=board; self.lock=lock or threading.Lock()
        self.running=False; self.thread=None; self.steps=0
        self.rng_key=jax.random.PRNGKey(int(time.time()))
        self.occ=scorer.make_occupancy(); self.jax_maps=scorer.make_maps()
        self.chunk_idx=0; 
        self.cnt={k:0 for k in ACTION_SET if k!='demoler'}
        self.active_mask={}
        self.blacklist=set()
        self.stuck_steps=0
        
        self.analista_terreno = AnalistaTerreno()
        self.heatmap_cache = None
        self.objetivo_actual = "Inicializando..."
        self.foco_construccion = None
        
        self.cache_suelos = [] 
        self.cache_decos = []

        self.agent=Policy(action_count=NA,obs_dim=3+NA,seed=42,hidden=128)
        try: self.agent.load('ml_data/policy/latest.ckpt')
        except: pass
        self._sync_full()

    def _sync_full(self):
        self.occ=scorer.make_occupancy()
        self.jax_maps=scorer.make_maps()
        self.cnt={k:0 for k in ACTION_SET if k!='demoler'}
        self.active_mask={}
        self.cache_suelos = [] 
        self.cache_decos = []
        self.blacklist.clear()

        for (x,y),v in self.board.items():
            if isinstance(v,tuple):
                bn=v[0]; w,h=_tam(bn)
                scorer.occ_mark_place(self.occ,x,y,w,h)
            elif isinstance(v,str):
                bn=v; w,h=1,1
                scorer.occ_mark_place(self.occ,x,y,1,1)
            
            if bn == 'suelo': self.cache_suelos.append((x,y))
            elif bn == 'decoracion': self.cache_decos.append((x,y))
            
            self._update_single_state(bn,x,y)

    def _update_single_state(self,bn,x,y):
        w,h=_tam(bn)
        coords=[(x+i,y+j) for i in range(w) for j in range(h)]
        is_active=R.es_activo(self.board,bn,coords)
        was_active=self.active_mask.get((x,y),False)
        if is_active and not was_active:
            self.jax_maps=scorer.apply_kernel_diff(self.jax_maps,bn,x,y,1.0)
            self.cnt[bn]=self.cnt.get(bn,0)+1
            self.active_mask[(x,y)]=True
        elif not is_active and was_active:
            self.jax_maps=scorer.apply_kernel_diff(self.jax_maps,bn,x,y,-1.0)
            self.cnt[bn]=max(0,self.cnt.get(bn,0)-1)
            self.active_mask[(x,y)]=False

    def _trigger_neighbors(self,x,y,w,h):
        adj=set()
        for i in range(-1,w+1):
            for j in range(-1,h+1):
                px,py=x+i,y+j
                if (px,py) in self.board:
                    v=self.board[(px,py)]
                    if isinstance(v,str): adj.add((px,py,v))
                    elif isinstance(v,tuple): adj.add((px,py,v[0]))
        for (ax,ay,abn) in adj:
            self._update_single_state(abn,ax,ay)

    def get_inactive_ids(self):
        with self.lock:
            s = set()
            for (x,y), active in self.active_mask.items():
                if not active: 
                    v = self.board.get((x,y))
                    if v:
                        if isinstance(v, tuple): s.add(v[1])
                        else: s.add((x,y))
            return s

    def get_heatmap(self):
        if self.heatmap_cache is None: return None
        matriz = np.array(self.heatmap_cache, dtype=np.float32)
        if np.max(matriz) > 0:
            matriz = matriz / np.max(matriz)
        return matriz

    def _get_full_stats(self):
        res = 0; jobs = 0; ind = 0; com = 0
        bal = {'energia': 0, 'agua': 0, 'comida': 0, 'basura': 0}
        active_count = sum(self.active_mask.values())
        total_count = 0
        
        for (x,y), v in self.board.items():
            bn = v[0] if isinstance(v, tuple) else str(v)
            if bn == 'suelo': continue
            total_count += 1
            
            if self.active_mask.get((x,y), False):
                if bn in STATS_CACHE:
                    d = STATS_CACHE[bn]
                    if d.energia != 0: bal['energia'] += d.energia
                    if d.agua != 0: bal['agua'] += d.agua
                    if d.comida != 0: bal['comida'] += d.comida
                    if d.basura != 0: bal['basura'] += d.basura
                    res += d.residentes
                    jobs += d.empleos
                    t = str(getattr(d.tipo, 'value', d.tipo)).lower()
                    if 'industria' in t: ind += d.empleos
                    elif 'comercio' in t: com += d.empleos
        
        return res, jobs, ind, com, bal, active_count, total_count

    def _diagnosticar_necesidad(self):
        res,jobs,ind,com,bal,active,total = self._get_full_stats()
        
        MIN_RECURSO_SEGURO = 16 
        BUFFER = 50.0

        # --- PRIORIDAD 1: DÉFICIT CRÍTICO (Balance NEGATIVO) ---
        if bal['energia'] <= 0: return 'refineria'
        if bal['agua'] <= 0: return 'agua'
        if bal['comida'] <= 0: return 'lecheria'
        if bal['basura'] <= 0: return 'depuradora'

        # --- PRIORIDAD 2: CONTEO MÍNIMO (Infraestructura de supervivencia) ---
        if self.cnt.get('refineria', 0) < MIN_RECURSO_SEGURO: return 'refineria'
        if self.cnt.get('agua', 0) < MIN_RECURSO_SEGURO: return 'agua'
        if self.cnt.get('depuradora', 0) < MIN_RECURSO_SEGURO: return 'depuradora'
        
        # Lechería es críticamente baja
        if self.cnt.get('lecheria', 0) < 60: return 'lecheria' 
        
        # --- PRIORIDAD 3: DESBALANCE DE EMPLEO (Bloqueo de Residencia) ---
        # Si la proporción Empleos/Residentes es crítica, construir Empleos.
        if res > 0 and jobs / res < 0.6: 
            if self.cnt.get('taller_togas', 0) < 20: return 'taller_togas' 
            if self.cnt.get('herreria', 0) < 20: return 'herreria'

        # --- PRIORIDAD 4: DÉFICIT DE BUFFER (Balance cerca de 0) ---
        if bal['energia'] < BUFFER: return 'refineria'
        if bal['agua'] < BUFFER: return 'agua'
        if bal['comida'] < BUFFER: return 'lecheria'
        if bal['basura'] < BUFFER: return 'depuradora'
        
        # --- PRIORIDAD FINAL: CRECIMIENTO ---
        return 'residencia'

    def _buscar_oportunidad_existente(self, bn):
        w, h = _tam(bn)
        req_suelo = R.requiere_suelo(bn)
        req_deco = R.requiere_decoracion(bn)
        
        fuente = []
        if req_suelo:
            if not self.cache_suelos: return False, None, None
            fuente = self.cache_suelos
        elif req_deco:
            if not self.cache_decos: return False, None, None
            fuente = self.cache_decos
        else:
            return False, None, None

        intentos = min(len(fuente), 50)
        muestras = random.sample(fuente, intentos)

        for sx, sy in muestras:
            for dx, dy in _TODOS8:
                vx, vy = sx + dx, sy + dy
                if 0 <= vx < S and 0 <= vy < S and (vx, vy) not in self.board:
                    cabe = True
                    if w > 1 or h > 1:
                        for i in range(w):
                            for j in range(h):
                                px, py = vx + i, vy + j
                                if not (0 <= px < S and 0 <= py < S) or (px, py) in self.board:
                                    cabe = False
                                    break
                            if not cabe: break
                    
                    if cabe:
                        cumple_todo = True
                        coords_edif = [(vx+i, vy+j) for i in range(w) for j in range(h)]
                        
                        if req_suelo and req_deco:
                            if not R.hay_tipo_adjacente(self.board, coords_edif, 'decoracion', True):
                                cumple_todo = False
                        
                        if req_deco and req_suelo:
                            if not R.hay_tipo_adjacente(self.board, coords_edif, 'suelo', True):
                                cumple_todo = False

                        if cumple_todo:
                            return True, vx, vy
                         
        return False, None, None

    def _verificar_capas_y_resolver(self, bn, x, y):
        w, h = _tam(bn)
        coords_for_check = [(x+i, y+j) for i in range(w) for j in range(h)]
        total_edificios = self._get_full_stats()[6]
        
        # --- CAPA 1: SUELO (Dependencia) ---
        if R.requiere_suelo(bn):
            necesita_suelo = not R.hay_tipo_adjacente(self.board, coords_for_check, 'suelo', True)
            
            if necesita_suelo:
                # *** VERIFICACIÓN DE RECURSOS (BLOQUEO DE RESIDENCIA) ***
                if bn == 'residencia':
                    stats = self._get_full_stats()
                    bal = stats[4]
                    is_critical_deficit = (bal['energia'] <= 0 or bal['agua'] <= 0 or 
                                           bal['comida'] <= 0 or bal['basura'] <= 0)
                    
                    if is_critical_deficit:
                        print(f"[RECURSOS] Detectado DÉFICIT CRÍTICO. Bloqueando la construcción de '{bn}' en ({x},{y}).")
                        return False # Falla la colocación, forzando la liberación del foco.
                
                
                # *** LOG DE PREGUNTA SÓLO SI NO HAY DÉFICIT Y SE NECESITA SUELO ***
                print(f"[PREGUNTA] Voy a ubicar el edificio '{bn}' en ({x},{y}). ¿Le falta SUELO para empezar? SÍ.")
                vecinos = self._obtener_vecinos_totales(x, y, w, h)
                
                # --- LÓGICA DE BYPASS DE DISTANCIA PARA AVANCE (CRÍTICO O CRECIMIENTO) ---
                es_critico_o_crecimiento = bn in ['refineria', 'agua', 'lecheria', 'depuradora', 'residencia']
                
                for nx, ny in vecinos:
                    if (nx,ny) not in self.board:
                        
                        validar_distancia = R.validar_distancia_suelo(self.board, nx, ny)
                        
                        # Aplicar el bypass si es un objetivo clave y la restricción de distancia bloquea
                        if es_critico_o_crecimiento and not validar_distancia:
                            validar_distancia = True 

                        if validar_distancia:
                            print(f"[ACCION] ¿Puedo ubicar SUELO en ({nx},{ny}) para satisfacer la dependencia de '{bn}'? SÍ.")
                            self._act_fast('suelo', nx, ny)
                            return True
                return False

        # --- CAPA 2: DECORACION (Dependencia) ---
        if R.requiere_decoracion(bn):
            if not R.hay_tipo_adjacente(self.board, coords_for_check, 'decoracion', True):
                print(f"[PREGUNTA] Voy a ubicar el edificio '{bn}' en ({x},{y}). ¿Le falta DECORACIÓN para empezar? SÍ.")
                vecinos = self._obtener_vecinos_totales(x, y, w, h)
                for nx, ny in vecinos:
                    if (nx,ny) not in self.board:
                        print(f"[ACCION] ¿Puedo ubicar DECORACIÓN en ({nx},{ny}) para satisfacer la dependencia de '{bn}'? SÍ.")
                        self._act_fast('decoracion', nx, ny)
                        return True
                return False

        # --- CAPA RAIZ: EL EDIFICIO MISMO (Finalidad) ---
        if (x,y) not in self.board:
            print(f"[PREGUNTA] ¿Ya resolví todas las dependencias? SÍ. ¿Puedo ubicar el edificio final '{bn}' en ({x},{y})? SÍ.")
            
            if not self._act_fast(bn, x, y):
                 # Si _act_fast falla aquí, es por colisión
                 return False
            
            return True
            
        return False

    def _obtener_vecinos_totales(self, x, y, w, h):
        vecinos = set()
        for i in range(-1, w+1):
            for j in range(-1, h+1):
                if 0 <= i < w and 0 <= j < h: continue 
                nx, ny = x + i, y + j
                if 0 <= nx < S and 0 <= ny < S:
                    vecinos.add((nx, ny))
        return list(vecinos)

    def _act_fast(self,bn,x,y):
        # El chequeo de ocupación debe ser para todas las celdas del edificio.
        w, h = _tam(bn)
        for i in range(w):
            for j in range(h):
                if (x + i, y + j) in self.board:
                    return False
        
        # Marcado de todas las celdas para la visualización y las colisiones.
        for i in range(w):
            for j in range(h):
                self.board[(x + i, y + j)] = bn # Ocupa todas las celdas.

        if bn == 'suelo': self.cache_suelos.append((x,y))
        elif bn == 'decoracion': self.cache_decos.append((x,y))
            
        scorer.occ_mark_place(self.occ,x,y,w,h)
        self._update_single_state(bn,x,y)
        self._trigger_neighbors(x,y,w,h)
        return True

    def simulation_step(self):
        with self.lock:
            # 1. VERIFICACIÓN Y DIAGNÓSTICO: SIEMPRE PRIMERO
            target_bn = self._diagnosticar_necesidad()
            self.objetivo_actual = target_bn
            
            hizo_algo = False
            
            # --- VERIFICACIÓN DE PRIORIDAD CRÍTICA Y ANULACIÓN DE FOCO ---
            stats = self._get_full_stats()
            bal = stats[4]
            is_critical_deficit = (bal['energia'] <= 0 or bal['agua'] <= 0 or 
                                   bal['comida'] <= 0 or bal['basura'] <= 0)
            
            # Si el Bot eligió Residencia (o está enfocado en ella), pero el balance es CRÍTICO, forzar la prioridad.
            if self.objetivo_actual == 'residencia' and is_critical_deficit:
                
                # Re-diagnóstico forzado basado en la realidad visible (IMAGEN)
                if bal['energia'] <= 0: target_bn = 'refineria'
                elif bal['agua'] <= 0: target_bn = 'agua'
                elif bal['comida'] <= 0: target_bn = 'lecheria'
                elif bal['basura'] <= 0: target_bn = 'depuradora'
                self.objetivo_actual = target_bn
                
            # Si el foco actual es RESIDENCIA y hay déficit, liberarlo inmediatamente.
            if self.foco_construccion and self.foco_construccion[0] == 'residencia' and is_critical_deficit:
                 self.foco_construccion = None
                 # La impresión de este log ahora ocurre en la capa de suelo si pasa el primer check
                 # print(f"[CRÍTICO] Foco de RESIDENCIA en {self.foco_construccion[1:] if self.foco_construccion else 'N/A'} anulado por DÉFICIT. Redirigiendo a {target_bn}.")


            # 2. MANEJO DE FOCO ACTUAL (Subordinado a la necesidad crítica)
            if self.foco_construccion:
                f_bn, f_x, f_y = self.foco_construccion
                
                w_foco, h_foco = _tam(f_bn)
                coords = [(f_x+i, f_y+j) for i in range(w_foco) for j in range(h_foco)]
                
                # Si el foco no es el recurso más crítico, LO DESTRUIMOS.
                if f_bn != target_bn:
                    self.foco_construccion = None
                else:
                    
                    if (f_x, f_y) in self.board:
                        v = self.board[(f_x,f_y)]
                        real = v[0] if isinstance(v,tuple) else str(v)
                        
                        if real == f_bn:
                            hizo_algo = self._verificar_capas_y_resolver(f_bn, f_x, f_y)
                            
                            if hizo_algo:
                                self.steps += 1
                                
                            # ESCAPE CRÍTICO 1: Si falló al resolver la capa, forzamos la liberación del foco.
                            elif not hizo_algo and not R.es_activo(self.board, f_bn, coords):
                                self.foco_construccion = None
                                # Marcar como fallido para no volver a intentar en el mismo lugar
                                self.blacklist.add((f_x, f_y))
                                
                            # Si el edificio ya está completo y activo, liberar el foco.
                            if R.es_activo(self.board, f_bn, coords):
                                 self.foco_construccion = None
                            
                        else:
                            self.foco_construccion = None
                    else:
                        hizo_algo = self._verificar_capas_y_resolver(f_bn, f_x, f_y)
                        if hizo_algo: 
                             self.steps += 1
                        # ESCAPE CRÍTICO 2: Si falló al resolver la capa al inicio, liberamos el foco.
                        elif not hizo_algo:
                            self.foco_construccion = None
                            # Marcar como fallido para no volver a intentar en el mismo lugar
                            self.blacklist.add((f_x, f_y))

            # Si se realizó una acción, terminamos el frame para re-diagnosticar en el siguiente.
            if hizo_algo:
                return

            # 3. ESTRATEGIA 1: OPORTUNIDAD (Busca espacio para el target_bn actual)
            if self.foco_construccion is None:
                # Limpiar blacklist si cambiamos de target_bn
                if self.objetivo_actual != target_bn:
                     self.blacklist.clear()
                     
                found, ox, oy = self._buscar_oportunidad_existente(target_bn)
                if found and (ox, oy) not in self.blacklist:
                    self.foco_construccion = (target_bn, ox, oy)
                    self._verificar_capas_y_resolver(target_bn, ox, oy)
                    self.steps += 1
                    return # Acción realizada, terminamos el frame.

            # 4. ESTRATEGIA 2: EXPANSIÓN (HEATMAP)
            if self.foco_construccion is None:
                rect = _get_chunk_rect(self.chunk_idx)
                desire_map = self.analista_terreno.generar_mapa_tactico(
                    target_bn, self.occ, self.jax_maps, rect
                )
                self.heatmap_cache = desire_map
                
                np_desire = np.array(desire_map)
                flat_indices = np.argsort(np_desire.ravel())[::-1]
                
                encontrado = False
                
                # AUMENTO DE VELOCIDAD DE DESCARTE (100)
                for idx in flat_indices[:100]: 
                    bx, by = divmod(int(idx), S)
                    
                    # Ignorar coordenadas fallidas
                    if (bx, by) in self.blacklist:
                        continue 

                    # Chequeo de deseo (solo si no es un recurso crítico)
                    if np_desire[bx, by] < 0.01 and target_bn == 'residencia': break
                    
                    # Intentar ubicación de capas
                    hizo = self._verificar_capas_y_resolver(target_bn, bx, by)
                    if hizo:
                        self.foco_construccion = (target_bn, bx, by)
                        encontrado = True
                        break
                    else:
                        # Si falla al intentar resolver la capa, la coordenada se añade a la blacklist en el escape.
                        pass
                
                if not encontrado and not self.foco_construccion:
                     self.stuck_steps += 1 
                     if self._check_unlock_conditions(): 
                         if self.stuck_steps > 30: 
                            if self.chunk_idx<(S//CHUNK_SIZE)**2-1:
                                self.chunk_idx+=1
                                self.blacklist.clear() # Limpiar blacklist al cambiar de lote
                                self.stuck_steps = 0 
                                print(f"[AUTO] >>> LOTE {self.chunk_idx} DESBLOQUEADO <<<")

            self.steps+=1
            if self.steps%50==0:
                print(f"[AUTO] St:{self.steps} Lote:{self.chunk_idx} Obj:{target_bn}")

    def _check_unlock_conditions(self):
        res,jobs,ind,com,bal,active,total=self._get_full_stats()
        
        # CRÍTICO: No desbloquear si hay déficit de recursos.
        if bal['energia'] < 0: return False
        if bal['agua'] < 0: return False
        if bal['comida'] < 0: return False
        if bal['basura'] < 0: return False

        lote_capacity = CHUNK_SIZE * CHUNK_SIZE
        capacity_total = (self.chunk_idx + 1) * lote_capacity
        if active < capacity_total * 0.75: 
            return False 

        if res > 10 and jobs < res * 0.9: 
             return False
        
        return True

    def _run(self):
        print(f"[AUTO] Gestor Logico Iniciado.")
        while self.running:
            try: self.simulation_step()
            except Exception as e: print(f"[ERR] {e}")
            time.sleep(0.001)

    def actualizar(self,st,sp):
        if st and not self.running:
            self.running=True
            self.thread=threading.Thread(target=self._run,daemon=True)
            self.thread.start()
        elif not st and self.running:
            self.running=False
            if self.thread: self.thread.join(timeout=0.2)
        return 0