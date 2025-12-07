# modo_automatico.py
from __future__ import annotations
import os,time,threading,jax,math
import jax.numpy as jnp
from config import BOARD_SIZE
import edificios,bd_celdas
from ml_policy import Policy
import scorer_jax as scorer
import restricciones as R

S=int(BOARD_SIZE)
ACTION_SET=('residencia','taller_togas','herreria','refineria','lecheria','agua',
            'depuradora','decoracion','policia','bombero','colegio','hospital',
            'suelo','demoler')
IDX={a:i for i,a in enumerate(ACTION_SET)}
NA=len(ACTION_SET)
BN_TO_ID={k:i+1 for i,k in enumerate(ACTION_SET) if k!='demoler'}; BN_TO_ID['suelo']=0

STATS_CACHE={k:edificios.edificios[k] for k in edificios.edificios if k in ACTION_SET and k!='demoler' and k!='suelo'}
CURR_SIZE=20; CURR_GROWTH=5; SCORE_GOAL=80.0
CHUNK_SIZE=20
SERVICE_RADIUS={'policia':15,'bombero':15,'colegio':12,'hospital':12,'decoracion':6}

class RealtimeStats:
    def __init__(self):
        self.totalResidentes=0; self.totalEmpleos=0
        self.totalEmpleosIndustria=0; self.totalEmpleosComercio=0
        self.porcentajeIndustria=0.0; self.porcentajeComercio=0.0
        self.desequilibrioLaboral=0
        self.energiaTotal=0; self.energiaUsada=0
        self.aguaTotal=0; self.aguaUsada=0
        self.comidaTotal=0; self.comidaUsada=0
        self.basuraTotal=0; self.basuraUsada=0
        self.ecologiaTotal=0; self.felicidadTotal=0

def _get_chunk_rect(idx):
    chunks_per_row=S//CHUNK_SIZE
    row=idx//chunks_per_row; col=idx%chunks_per_row
    x0,y0=col*CHUNK_SIZE,row*CHUNK_SIZE
    return x0,y0,x0+CHUNK_SIZE,y0+CHUNK_SIZE

def _obs_fast(maps,counts,score=0.0):
    s=jnp.sum; m=maps
    v=[float(s(m['seguridad']))/1000., float(s(m['felicidad']))/1000.]
    c=[float(counts.get(k,0))/50. for k in ACTION_SET]
    return jnp.array(v+c+[score/100.],dtype=jnp.float32)[:3+NA]

def _rew_fast(maps):
    s=jnp.sum; m=maps
    return float(s(m['felicidad'])*1.0+s(m['seguridad'])*0.5+s(m['ambiente'])*0.2)

def _tam(bn): return map(int,ED[bn].tamanio) if bn in ED else(1,1)
ED=edificios.edificios

class ModoAutomatico:
    __slots__=('board','running','thread','lock','agent','occ','jax_maps','steps',
               'rng_key','buf_o','buf_a','buf_r','buf_u','buf_v',
               'chunk_idx','last_score','baseline','cnt','active_mask',
               'last_pos','rep_count','blacklist','latest_ui_stats')
    
    def __init__(self,board,lock=None):
        self.board=board; self.lock=lock or threading.Lock()
        self.running=False; self.thread=None; self.steps=0
        self.rng_key=jax.random.PRNGKey(int(time.time()))
        self.occ=scorer.make_occupancy(); self.jax_maps=scorer.make_maps()
        self.chunk_idx=0; self.last_score=0.0; self.baseline=0.0
        self.cnt={k:0 for k in ACTION_SET if k!='demoler'}
        self.active_mask={}
        self.last_pos=None; self.rep_count=0; self.blacklist=set()
        self.latest_ui_stats = RealtimeStats()
        self.agent=Policy(action_count=NA,obs_dim=3+NA,seed=42,hidden=128)
        self.buf_o=[]; self.buf_a=[]; self.buf_r=[]; self.buf_u=[]; self.buf_v=[]
        try: self.agent.load(POLICY_ROOT)
        except: pass
        self._sync_full()

    def _sync_full(self):
        self.occ=scorer.make_occupancy()
        self.jax_maps=scorer.make_maps()
        self.cnt={k:0 for k in ACTION_SET if k!='demoler'}
        self.active_mask={}
        for (x,y),v in self.board.items():
            if isinstance(v,tuple):
                bn=v[0]; w,h=_tam(bn)
                scorer.occ_mark_place(self.occ,x,y,w,h)
            elif isinstance(v,str):
                scorer.occ_mark_place(self.occ,x,y,1,1)
        for (x,y),v in self.board.items():
            bn=v[0] if isinstance(v,tuple) else str(v)
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

    def _apply_labor_constraints(self):
        available_workers = 0
        job_buildings = []
        for (x,y), active in self.active_mask.items():
            if not active: continue
            v = self.board[(x,y)]; bn = v[0] if isinstance(v,tuple) else str(v)
            if bn == 'residencia':
                available_workers += STATS_CACHE[bn].residentes
            elif bn in STATS_CACHE:
                t = str(STATS_CACHE[bn].tipo).lower()
                if 'industria' in t or 'comercio' in t:
                    job_buildings.append(((x,y), bn, STATS_CACHE[bn].empleos))
        
        for (pos, bn, jobs_needed) in job_buildings:
            if available_workers >= jobs_needed:
                available_workers -= jobs_needed
            else:
                if self.active_mask.get(pos):
                    x,y = pos
                    self.jax_maps = scorer.apply_kernel_diff(self.jax_maps, bn, x, y, -1.0)
                    self.cnt[bn] = max(0, self.cnt.get(bn,0)-1)
                    self.active_mask[pos] = False

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

    def _calculate_stats_for_ui(self):
        s = RealtimeStats()
        for (x,y), active in self.active_mask.items():
            if not active: continue
            v = self.board[(x,y)]; bn = v[0] if isinstance(v,tuple) else str(v)
            if bn in STATS_CACHE:
                d = STATS_CACHE[bn]
                s.totalResidentes += d.residentes
                s.totalEmpleos += d.empleos
                s.energiaTotal += max(0, d.energia); s.energiaUsada += min(0, d.energia)
                s.aguaTotal += max(0, d.agua);       s.aguaUsada += min(0, d.agua)
                s.comidaTotal += max(0, d.comida);   s.comidaUsada += min(0, d.comida)
                s.basuraTotal += max(0, d.basura);   s.basuraUsada += min(0, d.basura)
                s.ecologiaTotal += d.ambiente
                s.felicidadTotal += d.felicidad
                t = str(d.tipo.value).lower() if hasattr(d.tipo,'value') else str(d.tipo).lower()
                if t=='industria': s.totalEmpleosIndustria += d.empleos
                elif t=='comercio': s.totalEmpleosComercio += d.empleos
        s.desequilibrioLaboral = s.totalResidentes - s.totalEmpleos
        if s.totalEmpleos > 0:
            s.porcentajeIndustria = (s.totalEmpleosIndustria / s.totalEmpleos) * 100
            s.porcentajeComercio = (s.totalEmpleosComercio / s.totalEmpleos) * 100
        self.latest_ui_stats = s

    def get_ui_data(self):
        with self.lock: return self.latest_ui_stats

    def _get_full_stats(self):
        res=0; jobs=0; ind=0; com=0
        bal={'energia':0,'agua':0,'comida':0,'basura':0}
        active_count=sum(self.active_mask.values())
        total_count=0
        for k,v in self.board.items():
            if isinstance(v,tuple) or (isinstance(v,str) and v!='suelo'):
                total_count+=1
        for (x,y),active in self.active_mask.items():
            if not active: continue
            v=self.board[(x,y)]; bn=v[0] if isinstance(v,tuple) else str(v)
            if bn in STATS_CACHE:
                d=STATS_CACHE[bn]
                res+=d.residentes; jobs+=d.empleos
                bal['energia']+=d.energia; bal['agua']+=d.agua
                bal['comida']+=d.comida; bal['basura']+=d.basura
                t=str(d.tipo.value).lower() if hasattr(d.tipo,'value') else str(d.tipo).lower()
                if t=='industria': ind+=d.empleos
                elif t=='comercio': com+=d.empleos
        return res,jobs,ind,com,bal,active_count,total_count

    def _find_fix_spot(self,tx,ty,w,h,allow_demolish=False):
        x0,y0,x1,y1=_get_chunk_rect(self.chunk_idx)
        candidates=[]
        for i in range(-2,w+2):
            for j in range(-2,h+2):
                nx,ny=tx+i,ty+j
                if nx<0 or nx>=S or ny<0 or ny>=S: continue
                if (nx,ny)==(tx,ty): continue
                if (nx,ny) in self.blacklist: continue
                penalty=0
                occupied=(nx,ny) in self.board
                if occupied:
                    if not allow_demolish: continue
                    v=self.board[(nx,ny)]
                    bn_n=v[0] if isinstance(v,tuple) else str(v)
                    if bn_n in ('agua','lecheria','refineria','depuradora','policia','bombero','colegio','hospital'):
                        continue
                    penalty+=500
                in_chunk=(x0<=nx<x1 and y0<=ny<y1)
                if not in_chunk: penalty+=1000
                candidates.append((penalty,nx,ny,occupied))
        if candidates:
            candidates.sort()
            best=candidates[0]
            if best[3]: return 'demolish',best[1],best[2]
            return 'build',best[1],best[2]
        return None,None,None

    def _find_spot_in_chunk(self,w_need,h_need):
        x0,y0,x1,y1=_get_chunk_rect(self.chunk_idx)
        best_demolish=None
        for x in range(x0,x1-w_need+1):
            for y in range(y0,y1-h_need+1):
                if (x,y) in self.blacklist: continue
                is_free=True
                for i in range(w_need):
                    for j in range(h_need):
                        if (x+i,y+j) in self.board:
                            is_free=False
                            v=self.board[(x+i,y+j)]
                            bn=v[0] if isinstance(v,tuple) else str(v)
                            if bn not in ('agua','lecheria','refineria','depuradora','policia','bombero','colegio','hospital'):
                                best_demolish=(x+i,y+j)
                            break
                    if not is_free: break
                if is_free: return 'build',x,y
        if best_demolish:
            return 'demolish',best_demolish[0],best_demolish[1]
        return None,None,None

    def _check_survival(self):
        inactives=[k for k,v in self.active_mask.items() if not v]
        
        is_crisis_overload = len(inactives) >= 10
        
        if inactives:
            for (x,y) in inactives:
                v=self.board.get((x,y))
                if not v: continue
                bn=v[0] if isinstance(v,tuple) else str(v)
                if bn in ('suelo','demoler'): continue
                
                is_job = False
                if bn in STATS_CACHE:
                    t = str(STATS_CACHE[bn].tipo).lower()
                    if 'industria' in t or 'comercio' in t: is_job = True
                
                w,h=_tam(bn)
                coords=[(x+i,y+j) for i in range(w) for j in range(h)]
                req_suelo = R.requiere_suelo(bn)
                req_decor = R.requiere_decoracion(bn)
                miss_suelo = req_suelo and not R.hay_tipo_adjacente(self.board,coords,'suelo',True)
                miss_decor = req_decor and not R.hay_tipo_adjacente(self.board,coords,'decoracion',True)
                
                if not miss_suelo and not miss_decor and is_job:
                    nw,nh = _tam('residencia')
                    mode,tx,ty = self._find_spot_in_chunk(nw,nh)
                    if mode=='build': return 'residencia',tx,ty
                    elif mode=='demolish': return 'demoler',tx,ty
                    
                    if is_crisis_overload:
                        break
                    continue 

                fix_bn=None
                if miss_suelo: fix_bn='suelo'
                elif miss_decor: fix_bn='decoracion'
                if fix_bn:
                    is_crit=bn in ('agua','lecheria','refineria','depuradora')
                    action,fx,fy=self._find_fix_spot(x,y,w,h,allow_demolish=is_crit)
                    if action=='build': return fix_bn,fx,fy
                    elif action=='demolish': return 'demoler',fx,fy
                    elif action is None and not is_crit: return 'demoler',x,y

                if is_crisis_overload and not fix_bn:
                    break

        res,jobs,ind,com,bal,active_count,total_count=self._get_full_stats()
        fixes={'agua':'agua','comida':'lecheria','energia':'refineria','basura':'depuradora'}
        for res,val in bal.items():
            if val<0:
                needed_bn=fixes[res]
                nw,nh=_tam(needed_bn)
                mode,tx,ty=self._find_spot_in_chunk(nw,nh)
                if mode=='build': return needed_bn,tx,ty
                elif mode=='demolish': return 'demoler',tx,ty

        srv_locs={k:[] for k in SERVICE_RADIUS}
        residences=[]
        for (x,y),active in self.active_mask.items():
            if active:
                v=self.board[(x,y)]; bn=v[0] if isinstance(v,tuple) else str(v)
                if bn in srv_locs: srv_locs[bn].append((x,y))
                elif bn=='residencia': residences.append((x,y))
        
        for rx,ry in residences:
            for s_type,s_rad in SERVICE_RADIUS.items():
                covered=False
                for sx,sy in srv_locs[s_type]:
                    if abs(rx-sx)+abs(ry-sy)<=s_rad:
                        covered=True; break
                if not covered:
                    nw,nh=_tam(s_type)
                    action,fx,fy=self._find_fix_spot(rx,ry,1,1,allow_demolish=False)
                    if action=='build': return s_type,fx,fy

        dec_locs=[]
        for (x,y),active in self.active_mask.items():
            if active:
                v=self.board[(x,y)]; bn=v[0] if isinstance(v,tuple) else str(v)
                if bn=='decoracion': dec_locs.append((x,y))
        
        for rx,ry in residences:
            covered=False
            for dx,dy in dec_locs:
                if abs(rx-dx)+abs(ry-dy)<=6:
                    covered=True; break
            if not covered:
                action,fx,fy=self._find_fix_spot(rx,ry,1,1,allow_demolish=False)
                if action=='build': return 'decoracion',fx,fy

        return None,None,None

    def _calc_complex_score(self):
        res,jobs,ind,com,bal,active,total=self._get_full_stats()
        score=float(res)*1.0
        score-=abs(res-jobs)*5.0
        score-=(total-active)*50.0
        if jobs>0:
            score-=(abs(ind/jobs - 1/3)+abs(com/jobs - 2/3))*200.0
        defic=sum(abs(v) for v in bal.values() if v<0)
        score-=defic*10.0
        return score

    def _act_fast(self,bn,x,y,force_replace=False):
        if bn != 'demoler' and bn != 'suelo':
            w,h = _tam(bn)
            coords = [(x+i, y+j) for i in range(w) for j in range(h)]
            if R.requiere_suelo(bn) and not R.hay_tipo_adjacente(self.board, coords, 'suelo', True):
                return False
            if R.requiere_decoracion(bn) and not R.hay_tipo_adjacente(self.board, coords, 'decoracion', True):
                return False

        if bn=='demoler':
            if (x,y) not in self.board: return False
            v=self.board.pop((x,y))
            old_bn=v[0] if isinstance(v,tuple) else str(v)
            if self.active_mask.get((x,y)):
                self.jax_maps=scorer.apply_kernel_diff(self.jax_maps,old_bn,x,y,-1.0)
                self.cnt[old_bn]=max(0,self.cnt.get(old_bn,0)-1)
                del self.active_mask[(x,y)]
            w,h=_tam(old_bn)
            scorer.occ_mark_clear(self.occ,x,y,w,h)
            bd_celdas.parchear_celda_en_disco(x,y,0)
            self._trigger_neighbors(x,y,w,h)
            return True
        elif bn=='suelo':
            if (x,y) in self.board and not force_replace: return False
            if not R.validar_distancia_suelo(self.board, x, y):
                return False
            
            if force_replace and (x,y) in self.board: self._act_fast('demoler',x,y)
            self.board[(x,y)]=bn
            w,h=_tam(bn)
            scorer.occ_mark_place(self.occ,x,y,w,h)
            self._update_single_state(bn,x,y)
            bd_celdas.parchear_celda_en_disco(x,y,BN_TO_ID.get(bn,0))
            self._trigger_neighbors(x,y,w,h)
            return True
        else:
            if (x,y) in self.board and not force_replace: return False
            if force_replace and (x,y) in self.board: self._act_fast('demoler',x,y)
            self.board[(x,y)]=bn
            w,h=_tam(bn)
            scorer.occ_mark_place(self.occ,x,y,w,h)
            self._update_single_state(bn,x,y)
            bd_celdas.parchear_celda_en_disco(x,y,BN_TO_ID.get(bn,0))
            self._trigger_neighbors(x,y,w,h)
            return True

    def simulation_step(self):
        with self.lock:
            self._apply_labor_constraints()
            self._calculate_stats_for_ui()
            cur_sc=self._calc_complex_score()
            obs=_obs_fast(self.jax_maps,self.cnt,cur_sc)
            help_bn,hx,hy=self._check_survival()
            if help_bn:
                if (hx,hy) == self.last_pos:
                    self.rep_count += 1
                else:
                    self.last_pos = (hx,hy)
                    self.rep_count = 0
                if self.rep_count > 10:
                    print(f"[AUTO] Bucle detectado en {hx},{hy}. Blacklisting.")
                    self.blacklist.add((hx,hy))
                    self.rep_count = 0
                    help_bn,hx,hy = self._check_survival()
            if help_bn and hx is not None:
                bn=help_bn
                tx,ty=hx,hy
                ai=IDX[bn]; u,v=float(hx)/S,float(hy)/S
                ok = self._act_fast(bn,tx,ty,force_replace=True)
            elif help_bn:
                ai,u,v,_=self.agent.act(obs); bn=ACTION_SET[ai]
                x0,y0,x1,y1=_get_chunk_rect(self.chunk_idx)
                tx=x0+int(u*(x1-x0)); ty=y0+int(v*(y1-y0))
                tx=max(x0,min(x1-1,tx)); ty=max(y0,min(y1-1,ty))
                ok = self._act_fast(bn,tx,ty,force_replace=False)
            else:
                ai,u,v,_=self.agent.act(obs); bn=ACTION_SET[ai]
                x0,y0,x1,y1=_get_chunk_rect(self.chunk_idx)
                tx=x0+int(u*(x1-x0)); ty=y0+int(v*(y1-y0))
                tx=max(x0,min(x1-1,tx)); ty=max(y0,min(y1-1,ty))
                ok = self._act_fast(bn,tx,ty,force_replace=False)
            prev=cur_sc
            curr=self._calc_complex_score()
            r=curr-prev
            if not ok: r -= 0.5 
            if self._check_unlock_conditions(): 
                if self.chunk_idx<(S//CHUNK_SIZE)**2-1:
                    self.chunk_idx+=1; r+=100.0
                    self.blacklist.clear()
                    print(f"[AUTO] CONDICIONES CUMPLIDAS! Desbloqueando Lote {self.chunk_idx}")
            self.buf_o.append(obs); self.buf_a.append(ai); self.buf_r.append(r)
            self.buf_u.append(float(u)); self.buf_v.append(float(v))
            if len(self.buf_r)>=128:
                adv=jnp.array(self.buf_r)-self.baseline
                self.agent.update(jnp.stack(self.buf_o),jnp.array(self.buf_a),jnp.array(self.buf_u),jnp.array(self.buf_v),adv,lr=2e-4)
                self.baseline=0.95*self.baseline+0.05*float(jnp.mean(jnp.array(self.buf_r)))
                self.buf_o.clear(); self.buf_a.clear(); self.buf_r.clear(); self.buf_u.clear(); self.buf_v.clear()
                if self.steps%1000==0: self.agent.save('ml_data/policy/latest.ckpt')
            self.steps+=1
            if self.steps%100==0:
                print(f"[AUTO] St:{self.steps} Lote:{self.chunk_idx} Sc:{curr:.1f}")

    def _check_unlock_conditions(self):
        res,jobs,ind,com,bal,active,total=self._get_full_stats()
        if total==0: return False
        if active/total < 0.99: return False
        for val in bal.values():
            if val < 0: return False
        if res < 50: return False
        if abs(res - jobs) > (res * 0.05): return False
        if jobs > 0:
            ind_ratio = ind / jobs
            if not (0.30 <= ind_ratio <= 0.36): return False
        else: return False
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