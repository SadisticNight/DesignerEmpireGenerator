# modo_automatico.py
from __future__ import annotations
import os,time,threading,atexit,signal
from typing import Any,Callable,Tuple,List
from array import array
import jax,jax.numpy as jnp
from config import BOARD_SIZE,LOG_COLOCACIONES,LOG_FALLOS,STEPWRITER_FLUSH_EVERY,SRV_CACHE_PERIOD,AUTO_TICK_SLEEP,AUTOSAVE_EVERY,AUTOSYNC_EVERY
import edificios,bd_celdas,areas,restricciones as R
from estadisticas import StatsProcessor
from ml_policy import Policy
from ml_replay import StepWriter
from generar_policy import save_policy_capnp as _save_policy_bin
import scorer_jax as scorer
def _noop(*_a,**_k):return 0
S=int(BOARD_SIZE)
ED=edificios.edificios
ACTION_SET='residencia','taller_togas','herreria','refineria','lecheria','agua','depuradora','decoracion','policia','bombero','colegio','hospital','suelo','demoler'
IDX={a:i for(i,a)in enumerate(ACTION_SET)}
NA=len(ACTION_SET)
SERVICIOS='policia','bombero','colegio','hospital'
SERV_IDX={k:i for(i,k)in enumerate(SERVICIOS)}
BLD_SERVICE_SCOPE=tuple(a for a in ACTION_SET if a not in('suelo','decoracion','demoler'))
POLICY_ROOT='ml_data/policy/latest.ckpt'
POLICY_BIN=POLICY_ROOT+'.bin'
_LOCKS=set()
_COV_FEATS=.0,.0,.0,.0
SRV_OVERLAP_FREE_ORIGINS=int(os.getenv('DEG_SRV_FREE_ORIGINS','4'))
K_SRV_OVERLAP_SOFT=float(os.getenv('DEG_K_SRV_OVERLAP_SOFT','-0.15'))
K_SRV_OVERLAP_HARD=float(os.getenv('DEG_K_SRV_OVERLAP_HARD','-0.80'))
K_ECO_DELTA=float(os.getenv('DEG_K_ECO_DELTA','0.03'))
K_FEL_DELTA=float(os.getenv('DEG_K_FEL_DELTA','0.03'))
BLD_COVER_MODE=os.getenv('DEG_BLD_COVER_MODE','any').lower()
K_BLD_COV_GAIN=float(os.getenv('DEG_K_BLD_COV_GAIN','1.0'))
K_BLD_UNCOV=float(os.getenv('DEG_K_BLD_UNCOV','-3.0'))
K_BLD_OVERLAP=float(os.getenv('DEG_K_BLD_OVERLAP','-0.2'))
K_DEMO_OVERLAP_GAIN=float(os.getenv('DEG_K_DEMO_OVERLAP_GAIN','5.0'))
CLUSTER_R=int(os.getenv('DEG_SRV_CLUSTER_R','8'))
REDUNDANT_UNIQUE_FRAC=float(os.getenv('DEG_REDUNDANT_UNIQUE_FRAC','0.08'))
MAINTENANCE_ON_FULL=int(os.getenv('DEG_MAINTENANCE','1'))!=0
MAINT_IDLE_OK=int(os.getenv('DEG_MAINT_IDLE','1'))!=0
TARGET_COV={'policia':float(os.getenv('DEG_TARGET_COV_POLICIA','0.70')),'bombero':float(os.getenv('DEG_TARGET_COV_BOMBERO','0.70')),'colegio':float(os.getenv('DEG_TARGET_COV_COLEGIO','0.70')),'hospital':float(os.getenv('DEG_TARGET_COV_HOSPITAL','0.70'))}
_DEG_RND_ON=int(os.getenv('DEG_RND','0'))!=0
_DEG_RND_SCALE=float(os.getenv('DEG_RND_SCALE','0.1'))
STATS_EVERY=int(os.getenv('DEG_STATS_EVERY','1'))
_GLOBAL_STEPS=0
_LAST_STATS=None
LOTES_ON=int(os.getenv('DEG_LOTES','1'))!=0
LOTE_SIZE=int(os.getenv('DEG_LOTE_SIZE','20'))
TOL_DESBAL=float(os.getenv('DEG_TOL_DESBAL','10'))
TOL_MIX=float(os.getenv('DEG_TOL_MIX','0.06'))
MIN_SURPLUS=float(os.getenv('DEG_MIN_SURPLUS','50'))
MIN_ECO=int(os.getenv('DEG_MIN_ECO','0'))
MIN_FEL=int(os.getenv('DEG_MIN_FEL','0'))
FORCE_FULL_COVER=int(os.getenv('DEG_FORCE_FULL_COVER','1'))!=0
NEG_ENV_DECOR_MIN_NEIGHBORS=int(os.getenv('DEG_DECOR_NEIGH_MIN','1'))
DECOR_BOOST=float(os.getenv('DEG_DECOR_BOOST','6.0'))
K_FAIL=-1.
K_RES_POS=+.1
K_RES_NEG=-.15
K_DESBAL=+.002
K_MIX=+.5
K_SRV_GAIN=+.5
K_SRV_OVERLAP=-.6
K_SRV_INSIDE=-.3
K_SUELO_ADJ=-.25
K_SOIL_BORDER=-.6
K_EDIF_SOIL_EXTRA=-.05
K_INACTIVE=-.4
K_SURPLUS=-.02
SURPLUS_THRESH=5e1
SURPLUS_SCALE=2e2
K_DEMOLER_BASE=-.2
STUCK_STEPS=800
MAX_SAME_FAILS=30
COOLDOWN_STEPS=600
RESAMPLE_TRIES=3
DEFICIT_TO_BN={'energia':'refineria','agua':'agua','comida':'lecheria','basura':'depuradora'}
PROV_TO_RES={'refineria':'energia','agua':'agua','lecheria':'comida','depuradora':'basura'}
_OFF8=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def _refresh_stats(board):
	global _LAST_STATS
	try:bd_celdas.update_celdas_bin(board,lock_resources=_LOCKS);bd_celdas.force_flush();_LAST_STATS=StatsProcessor().process()
	except Exception as e:print('[AUTO] fallo refresh stats:',repr(e))
def _tam(bn):w,h=ED[bn].tamanio;return int(w),int(h)
def _bloque(x,y,w,h):
	if x<0 or y<0 or x+w>S or y+h>S:return[]
	return[(x+i,y+j)for i in range(w)for j in range(h)]
def _vecinos8(c):
	x,y=c
	for(dx,dy)in _OFF8:
		nx,ny=x+dx,y+dy
		if 0<=nx<S and 0<=ny<S:yield(nx,ny)
def _es_borde(x,y):return x==0 or y==0 or x==S-1 or y==S-1
def _log_colocado(bn,coords):
	if LOG_COLOCACIONES:
		try:_sfx='...'if len(coords)>3 else'';print(f"Edificio {bn} construido en {list(coords)[:3]}{_sfx}")
		except Exception:pass
def _log_demolido(bn,iid):
	if LOG_FALLOS:
		try:print(f"Demolido {bn} (iid={iid})")
		except Exception:pass
def _extract_features(board):
	global _GLOBAL_STEPS,_LAST_STATS
	if _GLOBAL_STEPS%STATS_EVERY==0 or _LAST_STATS is None:_refresh_stats(board)
	s=_LAST_STATS;eg=max(0,-(s.energiaTotal+s.energiaUsada));ag=max(0,-(s.aguaTotal+s.aguaUsada));cg=max(0,-(s.comidaTotal+s.comidaUsada));bg=max(0,-(s.basuraTotal+s.basuraUsada));cnt=[0]*NA;seen=set()
	for v in board.values():
		if isinstance(v,tuple):
			bn,iid=v;key=str(bn).lower(),int(iid)
			if key in seen:continue
			seen.add(key)
			if key[0]in IDX:cnt[IDX[key[0]]]+=1
		else:
			bn=str(v).lower()
			if bn in IDX:cnt[IDX[bn]]+=1
	tot=max(1,sum(cnt));cnt=[c/tot for c in cnt];v=[s.totalResidentes,s.totalEmpleos,s.totalEmpleosIndustria,s.totalEmpleosComercio,s.porcentajeIndustria,s.porcentajeComercio,s.desequilibrioLaboral,eg,ag,cg,bg];norm=[1e5,1e5,1e5,1e5,100,100,1e3,1e3,1e3,1e3,1e3];v=[v[i]/norm[i]for i in range(len(v))];cov=list(_COV_FEATS);return jnp.array(v+cnt+cov,dtype=jnp.float32),s
def _ec_terms(s):te=float(s.totalEmpleos);tr=float(s.totalResidentes);des=float(abs(te-tr));ti=float(s.totalEmpleosIndustria);ratio=ti/max(te,1.);mix=float(abs(ratio-1./3.));deficits=[float(s.energiaTotal+s.energiaUsada),float(s.aguaTotal+s.aguaUsada),float(s.comidaTotal+s.comidaUsada),float(s.basuraTotal+s.basuraUsada)];neg=sum(-x for x in deficits if x<.0);return des,mix,neg
def _lock_from_stats(s):
	lr=set()
	if s.energiaTotal+s.energiaUsada>=0:lr.add('energia')
	if s.aguaTotal+s.aguaUsada>=0:lr.add('agua')
	if s.comidaTotal+s.comidaUsada>=0:lr.add('comida')
	if s.basuraTotal+s.basuraUsada>=0:lr.add('basura')
	return lr
def _needs_from_stats(s):
	need=set()
	if s.energiaTotal+s.energiaUsada<0:need.add('energia')
	if s.aguaTotal+s.aguaUsada<0:need.add('agua')
	if s.comidaTotal+s.comidaUsada<0:need.add('comida')
	if s.basuraTotal+s.basuraUsada<0:need.add('basura')
	return need
def _contar_suelos_ady_board(board,coords):
	seen=set();c=0
	for cell in coords:
		for nb in _vecinos8(cell):
			if nb in seen:continue
			if board.get(nb)=='suelo':seen.add(nb);c+=1
	return c
def _contar_suelos_alrededor_celda(board,pos):return sum(1 for nb in _vecinos8(pos)if board.get(nb)=='suelo')
def _surplus_penalty(s):
	xs=[float(s.energiaTotal+s.energiaUsada),float(s.aguaTotal+s.aguaUsada),float(s.comidaTotal+s.comidaUsada),float(s.basuraTotal+s.basuraUsada)];surplus=sum(max(.0,x-SURPLUS_THRESH)for x in xs)
	if surplus<=0:return .0
	return K_SURPLUS*(surplus/SURPLUS_SCALE)
def _sig_progress(s):return int(s.totalResidentes),int(s.totalEmpleos),int(s.energiaTotal+s.energiaUsada),int(s.aguaTotal+s.aguaUsada),int(s.comidaTotal+s.comidaUsada),int(s.basuraTotal+s.basuraUsada)
def _centroid_cell(cells):
	xs=[x for(x,_)in cells];ys=[y for(_,y)in cells]
	if not xs:return 0,0
	cx=int(round(sum(xs)/max(1,len(xs))));cy=int(round(sum(ys)/max(1,len(ys))));return max(0,min(S-1,cx)),max(0,min(S-1,cy))
class ModoAutomatico:
	__slots__='board','running','thread','updated','lock','agent','buf_o','buf_a','buf_u','buf_v','buf_r','baseline','writer','episode_id','steps','last_obs','last_stats','last_terms','last_lock_resources','last_need_resources','fails','game_over','_iid_counter','srv_union','srv_origins','_zona_cache','_srv_dirty','on_change','flush_every','cache_period','tick_sleep','last_change_step','last_progress_step','progress_sig','cooldown','same_fail_type','same_fail_count','_phi','rng_key','_in_maintenance','bld_foot','rnd','rnd_scale','_step_handlers','occ','srv_ref','srv_cov_counts','lotes_on','lote_size','lotes','unlocked_tiles','_demo_target','bld_counts'
	def __init__(self,board,flush_every=None,cache_period=None,tick_sleep=None):
		self.board=board;self.running=False;self.thread=None;self.updated=False;self.lock=threading.Lock();env_seed=int(os.getenv('DEG_ENV_SEED','1337'));self.rng_key=jax.random.PRNGKey(env_seed);_gi=lambda k,d:int(os.getenv(k))if os.getenv(k)else d;_gf=lambda k,d:float(os.getenv(k))if os.getenv(k)else d;self.flush_every=flush_every if flush_every is not None else _gi('DEG_FLUSH_EVERY',STEPWRITER_FLUSH_EVERY);self.cache_period=cache_period if cache_period is not None else _gi('DEG_SRV_CACHE',SRV_CACHE_PERIOD);self.tick_sleep=tick_sleep if tick_sleep is not None else _gf('DEG_TICK_SLEEP',AUTO_TICK_SLEEP);self.occ=scorer.make_occupancy();self.srv_union={k:set()for k in SERVICIOS};self.srv_origins={k:[]for k in SERVICIOS};self._zona_cache={};self._rebuild_srv_cache_full();self.bld_foot={};self._rebuild_bld_foot_full();self._rebuild_occ_full();self.srv_ref=array('H',[0])*(S*S*len(SERVICIOS));self.srv_cov_counts=[0,0,0,0];self._rebuild_srv_ref_full();self._init_action_context();self._update_cov_feats_global();self.lotes_on=LOTES_ON;self.lote_size=max(1,min(LOTE_SIZE,S));self._init_lotes();self._rebuild_bld_counts_full();obs0,s0=_extract_features(self.board);self.last_obs=obs0;self.last_stats=s0;self.last_terms=_ec_terms(s0);self.last_lock_resources=_lock_from_stats(s0);self.last_need_resources=_needs_from_stats(s0);global _LOCKS;_LOCKS=set(self.last_lock_resources);self.agent=Policy(action_count=NA,obs_dim=int(obs0.shape[0]),seed=42,hidden=128,entropy_coef=.007)
		try:self.agent.action_context=self._phi
		except Exception:pass
		try:
			ok=self.agent.load(POLICY_ROOT)
			if not ok and not os.path.exists(POLICY_BIN):self._save_ckpt()
		except Exception:
			if not os.path.exists(POLICY_BIN):self._save_ckpt()
		self.buf_o=[];self.buf_a=[];self.buf_u=[];self.buf_v=[];self.buf_r=[];self.baseline=.0;self.episode_id=time.perf_counter_ns()//1000000
		try:self.writer=StepWriter(episode_id=self.episode_id,version=3,flush_every=self.flush_every)
		except Exception:self.writer=None
		self.steps=0;self.fails=0;self.game_over=False;self._iid_counter=1;self._srv_dirty=False;self.on_change=None;self.last_change_step=0;self.last_progress_step=0;self.progress_sig=_sig_progress(s0);self.cooldown={};self.same_fail_type=None;self.same_fail_count=0;self._in_maintenance=False;self.rnd=None;self.rnd_scale=float(_DEG_RND_SCALE);self._demo_target=None
		if _DEG_RND_ON:
			try:from intrinsic_rnd import RND;self.rnd=RND(obs_dim=int(obs0.shape[0]),seed=env_seed);print('[AUTO] RND activado.')
			except Exception as e:print('[AUTO] RND no disponible:',repr(e));self.rnd=None
		self._step_handlers={'maint_idle':self._step_maint_idle,'maint_build':self._step_maint_build,'force_demo':self._step_force_demo,'act':self._step_act};self._install_traps()
	def _install_traps(self):
		def _graceful_exit(signum=None,frame=None):
			try:self.save_now()
			finally:
				try:print('[AUTO] salida por señal, guardado OK')
				except Exception:pass
				raise SystemExit(0)
		atexit.register(lambda:self.save_now())
		try:signal.signal(signal.SIGINT,_graceful_exit);signal.signal(signal.SIGTERM,_graceful_exit)
		except Exception:pass
	def _first_lote_rect(self):L=self.lote_size;cx,cy=S//2,S//2;h=L//2;x0,y0=cx-h,cy-h;x1,y1=x0+L,y0+L;return max(0,x0),max(0,y0),min(S,x1),min(S,y1)
	def _gen_lotes_spiral_centered(self):
		L=self.lote_size;x0,y0,x1,y1=self._first_lote_rect();rects=[(x0,y0,x1,y1)];dirs=(1,0),(0,1),(-1,0),(0,-1);step=1;x=y=0;max_rects=(S//L+2)*(S//L+2)
		while len(rects)<max_rects:
			for(di,(dx,dy))in enumerate(dirs):
				for _ in range(step):
					x+=dx;y+=dy;rx0=x0+x*L;ry0=y0+y*L;rx1=rx0+L;ry1=ry0+L;rx0,ry0=max(0,rx0),max(0,ry0);rx1,ry1=min(S,rx1),min(S,ry1)
					if rx0<rx1 and ry0<ry1:
						rect=rx0,ry0,rx1,ry1
						if rects[-1]!=rect:rects.append(rect)
				if di%2==1:step+=1
			if any(r[0]==0 for r in rects)and any(r[2]==S for r in rects)and any(r[1]==0 for r in rects)and any(r[3]==S for r in rects):break
		return rects
	def _init_lotes(self):self.lotes=self._gen_lotes_spiral_centered();self.unlocked_tiles=1 if LOTES_ON else len(self.lotes);r0=self.lotes[0];print(f"[LOTE] size={self.lote_size} primer={r0} desbloqueados={self.unlocked_tiles}/{len(self.lotes)}")
	def _is_allowed(self,x,y):
		if not self.lotes_on:return True
		for i in range(self.unlocked_tiles):
			x0,y0,x1,y1=self.lotes[i]
			if x0<=x<x1 and y0<=y<y1:return True
		return False
	def _rand_in_unlocked(self):
		if not self.lotes_on:return self._randint(0,S-1),self._randint(0,S-1)
		import random;i=random.randrange(self.unlocked_tiles);x0,y0,x1,y1=self.lotes[i];return self._randint(x0,max(x0,x1-1)+1),self._randint(y0,max(y0,y1-1)+1)
	def _maybe_unlock_next(self,s):
		if not self.lotes_on:return False
		if self.unlocked_tiles>=len(self.lotes):return False
		te=float(s.totalEmpleos);tr=float(s.totalResidentes);des_ok=abs(te-tr)<=TOL_DESBAL;ind=float(s.totalEmpleosIndustria);com=float(s.totalEmpleosComercio);te_safe=max(te,1.);mix_i_ok=abs(ind/te_safe-1./3.)<=TOL_MIX;mix_c_ok=abs(com/te_safe-2./3.)<=TOL_MIX;surpl=[float(s.energiaTotal+s.energiaUsada),float(s.aguaTotal+s.aguaUsada),float(s.comidaTotal+s.comidaUsada),float(s.basuraTotal+s.basuraUsada)];surplus_ok=all(v>=MIN_SURPLUS for v in surpl);eco_ok=int(s.ecologiaTotal)>=MIN_ECO;fel_ok=int(s.felicidadTotal)>=MIN_FEL;cov_total,unc=self._coverage_stats();cover_ok=sum(unc.values())==0 if FORCE_FULL_COVER else True;all_ok=des_ok and mix_i_ok and mix_c_ok and surplus_ok and eco_ok and fel_ok and cover_ok
		if all_ok:
			self.unlocked_tiles+=1;i=self.unlocked_tiles-1;x0,y0,x1,y1=self.lotes[i]
			try:print(f"[LOTE] Desbloqueado #{i+1}/{len(self.lotes)} -> ({x0},{y0})-({x1},{y1})")
			except Exception:pass
			return True
		return False
	def _rand(self):self.rng_key,sub=jax.random.split(self.rng_key);return float(jax.random.uniform(sub,()))
	def _randint(self,low,high):self.rng_key,sub=jax.random.split(self.rng_key);return int(jax.random.randint(sub,(),low,high))
	def _choice(self,seq):
		if not seq:return
		self.rng_key,sub=jax.random.split(self.rng_key);idx=int(jax.random.randint(sub,(),0,len(seq)));return seq[idx]
	def _init_action_context(self):
		def _absmax(vals):
			m=1.
			for v in vals:av=abs(float(v));m=av if av>m else m
			return m
		energia=[ED[b].energia for b in ED if b in ED];agua=[ED[b].agua for b in ED if b in ED];comida=[ED[b].comida for b in ED if b in ED];basura=[ED[b].basura for b in ED if b in ED];empleos=[ED[b].empleos for b in ED if b in ED];residentes=[ED[b].residentes for b in ED if b in ED];felicidad=[ED[b].felicidad for b in ED if b in ED];ambiente=[ED[b].ambiente for b in ED if b in ED];ws=[ED[b].tamanio[0]for b in ED if b in ED];hs=[ED[b].tamanio[1]for b in ED if b in ED];mE=_absmax(energia);mA=_absmax(agua);mC=_absmax(comida);mB=_absmax(basura);mJ=_absmax(empleos);mR=_absmax(residentes);mF=_absmax(felicidad);mAM=_absmax(ambiente);mW=max(1.,max(ws)if ws else 1.);mH=max(1.,max(hs)if hs else 1.);TIPOS='residencia','comercio','industria','decoracion','suelo';PROVIDE='energia','agua','comida','basura';SERV=SERVICIOS;cx=cy=S//2
		def zona_size(bn):
			try:return float(len(areas.Area.zona_cubierta_por_edificio(bn,(cx,cy),S)))
			except Exception:return .0
		denom_area=float(S*S)if S>0 else 1.;PHI=[]
		for a in ACTION_SET:
			if a not in ED:PHI.append([.0]*(10+len(TIPOS)+len(PROVIDE)+len(SERV)+3));continue
			e=ED[a];w,h=e.tamanio;base=[e.energia/mE,e.agua/mA,e.comida/mC,e.basura/mB,e.empleos/mJ,e.residentes/mR,e.felicidad/mF,e.ambiente/mAM,float(w)/mW,float(h)/mH];tname=str(getattr(e.tipo,'value',e.tipo)).lower();tvec=[1. if tname==t else .0 for t in TIPOS];pvec=[1. if getattr(e,PROVIDE[i])>0 else .0 for i in range(len(PROVIDE))];svec=[1. if a==srv else .0 for srv in SERV];area=zona_size(a)/denom_area;area_fel=area if e.felicidad>0 else .0;area_eco=area if e.ambiente>0 else .0;area_srv=area if a in SERVICIOS else .0;PHI.append(base+tvec+pvec+svec+[area_fel,area_eco,area_srv])
		self._phi=jnp.asarray(PHI,dtype=jnp.float32)
	def _zona(self,bn,pos):
		k=bn,pos[0],pos[1];z=self._zona_cache.get(k)
		if z is None:z=set(areas.Area.zona_cubierta_por_edificio(bn,pos,S));self._zona_cache[k]=frozenset(z)
		return set(self._zona_cache[k])
	def _rebuild_srv_cache_full(self):
		self.srv_union={k:set()for k in SERVICIOS};self.srv_origins={k:[]for k in SERVICIOS};seen={}
		for((x,y),v)in self.board.items():
			if isinstance(v,tuple):
				bn,iid=v
				if bn in SERVICIOS:
					if iid not in seen:seen[iid]=x,y
					else:ox,oy=seen[iid];seen[iid]=min(ox,x),min(oy,y)
		for((x,y),v)in self.board.items():
			if isinstance(v,tuple):
				bn,iid=v
				if bn in SERVICIOS and seen.get(iid)==(x,y):self.srv_origins[bn].append((x,y));self.srv_union[bn]|=self._zona(bn,(x,y))
	def _rebuild_bld_foot_full(self):
		self.bld_foot.clear()
		for((x,y),v)in self.board.items():
			if isinstance(v,tuple):
				bn,iid=v
				if bn in BLD_SERVICE_SCOPE:self.bld_foot.setdefault(int(iid),set()).add((x,y))
	def _rebuild_occ_full(self):
		occ=scorer.make_occupancy();origins={};singles=[]
		for((x,y),v)in self.board.items():
			if isinstance(v,tuple):bn,iid=v;origins.setdefault((bn,int(iid)),[]).append((x,y))
			else:singles.append((x,y))
		for((bn,iid),cells)in origins.items():minx=min(c[0]for c in cells);miny=min(c[1]for c in cells);w,h=_tam(bn);occ=scorer.occ_mark_place(occ,minx,miny,w,h)
		for(x,y)in singles:occ=scorer.occ_mark_place(occ,x,y,1,1)
		self.occ=occ
	def _rebuild_srv_ref_full(self):
		self.srv_ref[:]=array('H',[0])*(S*S*len(SERVICIOS));self.srv_cov_counts=[0,0,0,0];seen={}
		for((x,y),v)in self.board.items():
			if isinstance(v,tuple):
				bn,iid=v
				if bn in SERVICIOS:
					if iid not in seen:seen[iid]=x,y
					else:ox,oy=seen[iid];seen[iid]=min(ox,x),min(oy,y)
		for((x,y),v)in self.board.items():
			if isinstance(v,tuple):
				bn,iid=v
				if bn in SERVICIOS and seen.get(iid)==(x,y):self._srv_apply_zone(bn,x,y,+1)
	def _update_cov_feats_global(self):global _COV_FEATS;den=float(S*S)if S>0 else 1.;_COV_FEATS=tuple(self.srv_cov_counts[i]/den for i in range(len(SERVICIOS)))
	def _srv_apply_zone(self,bn,x,y,delta):
		k=SERV_IDX.get(bn,None)
		if k is None:return
		z=self._zona(bn,(x,y))
		if not z:return
		ref=self.srv_ref;cov=self.srv_cov_counts;L=len(SERVICIOS)
		for(cx,cy)in z:
			if self.lotes_on and not self._is_allowed(cx,cy):continue
			i=(cy*S+cx)*L+k;v=ref[i];nv=v+delta
			if nv<0:nv=0
			if v==0 and nv>0:cov[k]+=1
			elif v>0 and nv==0:cov[k]-=1
			ref[i]=nv
	def _bld_is_covered_by(self,cells,stype,mode):
		if not cells:return False
		k=SERV_IDX.get(stype,None)
		if k is None:return False
		L=len(SERVICIOS);ref=self.srv_ref
		if mode=='any':return any(ref[(y*S+x)*L+k]>0 for(x,y)in cells)
		if mode=='all':return all(ref[(y*S+x)*L+k]>0 for(x,y)in cells)
		cx,cy=_centroid_cell(cells);return ref[(cy*S+cx)*L+k]>0
	def _coverage_stats(self):
		unc={k:0 for k in SERVICIOS};total=0
		for(_,cells)in self.bld_foot.items():
			total+=1
			for st in SERVICIOS:
				if not self._bld_is_covered_by(cells,st,BLD_COVER_MODE):unc[st]+=1
		return total,unc
	def _shape_bld_coverage(self,pre_stats,post_stats,bn,x,y,ok):
		total0,unc0=pre_stats;total1,unc1=post_stats
		if total1<=0:return .0
		gain=sum(max(0,unc0[s]-unc1[s])for s in SERVICIOS);r=.0;r+=K_BLD_COV_GAIN*(gain/total1);r+=K_BLD_UNCOV*(sum(unc1.values())/total1)
		if ok and bn in SERVICIOS:
			zona=self._zona(bn,(x,y))
			if zona:k=SERV_IDX[bn];L=len(SERVICIOS);ref=self.srv_ref;overlap_same=sum(1 for(cx,cy)in zona if ref[(cy*S+cx)*L+k]>0);ofrac_same=overlap_same/float(len(zona));r+=K_BLD_OVERLAP*ofrac_same
		return float(r)
	def _emit_change(self,kind,bn,x,y,w,h):
		cb=self.on_change
		if cb:
			try:cb(kind,bn,x,y,w,h)
			except Exception:pass
		self.last_change_step=self.steps
	def _rebuild_bld_counts_full(self):
		cnt={};seen=set()
		for v in self.board.values():
			if isinstance(v,tuple):
				bn,iid=v;k=bn,int(iid)
				if k in seen:continue
				seen.add(k);cnt[bn]=cnt.get(bn,0)+1
			else:bn=str(v);cnt[bn]=cnt.get(bn,0)+1
		self.bld_counts=cnt
	def _provider_exists(self,bn):return self.bld_counts.get(bn,0)>0
	def _resample_action(self,probs,banned_idx):
		p=jnp.asarray([max(.0,float(x))if x==x else .0 for x in probs],jnp.float32)
		if banned_idx is not None and 0<=banned_idx<p.shape[0]:p=p.at[banned_idx].set(.0)
		tot=float(jax.device_get(jnp.sum(p)));p=jnp.ones_like(p)/p.shape[0]if tot<=.0 else p/tot;self.rng_key,sub=jax.random.split(self.rng_key);idxs=jnp.arange(p.shape[0]);return int(jax.random.choice(sub,idxs,p=p))
	def _ensure_prereqs(self,bn,x,y):
		if bn not in ED:return True
		if not self._is_allowed(x,y):return False
		w,h=_tam(bn);coords=_bloque(x,y,w,h)
		if not coords:return True
		if self.lotes_on and any(not self._is_allowed(cx,cy)for(cx,cy)in coords):return False
		ok=True;pre=('suelo',lambda:R.requiere_suelo(bn),lambda:R.hay_tipo_adjacente(self.board,coords,'suelo',True)),('decoracion',lambda:R.requiere_decoracion(bn),lambda:R.hay_tipo_adjacente(self.board,coords,'decoracion',True))
		for(kind,need_fn,has_adj_fn)in pre:
			try:need=need_fn()and not has_adj_fn()
			except Exception:need=False
			if ok and need:
				placed=False
				for(cx,cy)in coords:
					for nb in _vecinos8((cx,cy)):
						if nb not in coords and nb not in self.board and self._is_allowed(nb[0],nb[1]):
							if self._place(kind,nb[0],nb[1]):placed=True;break
					if placed:break
				if not placed:ok=False
		return ok
	def _coverage_deficits(self):
		gaps=[]
		for k in SERVICIOS:
			cov=float(_COV_FEATS[SERV_IDX[k]]);tgt=float(TARGET_COV[k])
			if cov+1e-06<tgt:gaps.append((k,tgt-cov))
		gaps.sort(key=lambda x:-x[1]);return gaps
	def _adjacent_to_cells(self,cells):
		adj=set()
		for(x,y)in cells:
			for nb in _vecinos8((x,y)):
				if nb not in cells and 0<=nb[0]<S and 0<=nb[1]<S:
					if not self.lotes_on or self._is_allowed(nb[0],nb[1]):adj.add(nb)
		return adj
	def _count_decor_adj(self,cells):return sum(1 for nb in self._adjacent_to_cells(cells)if self.board.get(nb)=='decoracion')
	def _neg_env_targets(self):
		cand=[]
		for((x,y),v)in self.board.items():
			if not isinstance(v,tuple):continue
			bn,iid=v
			try:amb=ED[bn].ambiente
			except Exception:amb=0
			if amb>=0:continue
			cells=self.bld_foot.get(int(iid))
			if not cells:cells={c for(c,val)in self.board.items()if val==(bn,iid)}
			if not cells:continue
			have=self._count_decor_adj(cells);need=max(0,NEG_ENV_DECOR_MIN_NEIGHBORS-have)
			if need<=0:continue
			libres=[p for p in self._adjacent_to_cells(cells)if p not in self.board]
			if not libres:continue
			weight=abs(int(amb))*need*max(1,len(cells))
			for p in libres:cand.append((p,weight))
		cand.sort(key=lambda kv:-kv[1]);return cand
	def _pick_decor_target(self):cand=self._neg_env_targets();return cand[0][0]if cand else None
	def _total_overlap_ratio(self):
		L=len(SERVICIOS);ref=self.srv_ref;dup=0;total=S*S*L
		for i in range(0,total,L):
			if ref[i+0]>=2:dup+=1
			if ref[i+1]>=2:dup+=1
			if ref[i+2]>=2:dup+=1
			if ref[i+3]>=2:dup+=1
		return float(dup)/float(total if total>0 else 1)
	def _same_service_neighbors(self,bn,x,y,r=CLUSTER_R):return sum(1 for(ox,oy)in self.srv_origins.get(bn,[])if abs(ox-x)+abs(oy-y)<=r)-1
	def _origin_overlap_stats(self,bn,x,y):
		k=SERV_IDX[bn];zona=self._zona(bn,(x,y))
		if not zona:return 0,0,.0
		L=len(SERVICIOS);ref=self.srv_ref;covered=0;unique=0
		for(cx,cy)in zona:
			v=ref[(cy*S+cx)*L+k]
			if v>0:
				covered+=1
				if v==1:unique+=1
		if covered==0:return 0,0,.0
		overlap_frac=1.-unique/float(covered);return covered,unique,overlap_frac
	def _pick_redundant_service_origin(self):
		best=None;best_score=.0
		for bn in SERVICIOS:
			for(x,y)in self.srv_origins.get(bn,[]):
				covered,unique,overlap_frac=self._origin_overlap_stats(bn,x,y)
				if covered<=0:continue
				unique_frac=unique/float(covered);neigh=self._same_service_neighbors(bn,x,y,CLUSTER_R)
				if unique_frac<=REDUNDANT_UNIQUE_FRAC and neigh>=1:
					score=overlap_frac*(1.+.25*neigh)
					if score>best_score:best_score=score;best=x,y
		return best
	def _neighbors_of_block(self,x,y,w,h):
		nb=set()
		for i in range(w):
			for j in(-1,h):
				xx,yy=x+i,y+j
				if 0<=xx<S and 0<=yy<S:nb.add((xx,yy))
		for j in range(h):
			for i in(-1,w):
				xx,yy=x+i,y+j
				if 0<=xx<S and 0<=yy<S:nb.add((xx,yy))
		return list(nb)
	def _clear_for_place(self,bn,x,y):
		if bn not in ED:return True
		if not self._is_allowed(x,y):return False
		w,h=_tam(bn);coords=_bloque(x,y,w,h)
		if not coords:return False
		if self.lotes_on and any(not self._is_allowed(cx,cy)for(cx,cy)in coords):return False
		seen_iids=set()
		for c in coords:
			v=self.board.get(c)
			def _t_tuple():
				b2,iid=v;key=b2,int(iid)
				if key not in seen_iids:self._demoler_en(c);seen_iids.add(key)
			def _t_other():self._demoler_en(c)
			{tuple:_t_tuple,str:_t_other}.get(type(v),_noop)()
		need_ok=self._ensure_prereqs(bn,x,y)
		if not need_ok:
			tried=0
			for nb in self._neighbors_of_block(x,y,w,h):
				if nb not in self.board:continue
				self._demoler_en(nb);tried+=1
				if tried>=2 or self._ensure_prereqs(bn,x,y):break
		return True
	def _new_iid(self):self._iid_counter+=1;return self._iid_counter
	def _place(self,bn,x,y):
		if not self._is_allowed(x,y):return False
		w,h=_tam(bn)
		if 0<=x<=S-w and 0<=y<=S-h:
			coords=_bloque(x,y,w,h)
			if self.lotes_on and any(not self._is_allowed(cx,cy)for(cx,cy)in coords):return False
			if coords and all(c not in self.board for c in coords):
				is_single=bn in('suelo','decoracion')and w==1 and h==1
				def _single():self.board[x,y]=bn;_log_colocado(bn,[(x,y)]);self.occ=scorer.occ_mark_place(self.occ,x,y,1,1);self.bld_counts[bn]=self.bld_counts.get(bn,0)+1
				def _multi():
					iid=self._new_iid();self.board.update({c:(bn,iid)for c in coords});_log_colocado(bn,coords);self.occ=scorer.occ_mark_place(self.occ,x,y,w,h);self.bld_counts[bn]=self.bld_counts.get(bn,0)+1
					if bn in BLD_SERVICE_SCOPE:self.bld_foot[int(iid)]=set(coords)
				{True:_single,False:_multi}[is_single]()
				if bn in SERVICIOS:self.srv_origins[bn].append((x,y));self.srv_union[bn]|=self._zona(bn,(x,y));self._srv_apply_zone(bn,x,y,+1);self._update_cov_feats_global()
				self._emit_change('placed',bn,x,y,w,h);_refresh_stats(self.board);return True
		return False
	def _try_place(self,bn,x,y):
		try:cand=scorer.best_spot(self.board,bn,{k:_tam(k)for k in ED},topk=512,occ=self.occ)
		except Exception:cand=None
		if cand and self._is_allowed(cand[0],cand[1]):
			cx,cy=cand
			if self._place(bn,cx,cy):return True
		if not self._is_allowed(x,y):x,y=self._rand_in_unlocked()
		if self._place(bn,x,y):return True
		w,h=_tam(bn)
		for r in(0,1,2,3):
			for dx in(-r,0,r):
				for dy in(-r,0,r):
					px,py=x+dx,y+dy
					if self._is_allowed(px,py)and self._place(bn,px,py):return True
		return False
	def _demoler_en(self,pos):
		x,y=pos
		if self.lotes_on and not self._is_allowed(x,y):return False
		v=self.board.get(pos)
		if v is None:return False
		if isinstance(v,tuple):
			bn,iid=v;minx=miny=10**9;maxx=maxy=-1
			for(c,val)in list(self.board.items()):
				if val==(bn,iid):
					xx,yy=c
					if xx<minx:minx=xx
					if yy<miny:miny=yy
					if xx>maxx:maxx=xx
					if yy>maxy:maxy=yy
					del self.board[c]
			_log_demolido(bn,iid)
			if bn in BLD_SERVICE_SCOPE:self.bld_foot.pop(int(iid),None)
			if bn in SERVICIOS and minx<=maxx and miny<=maxy:self._srv_apply_zone(bn,minx,miny,-1)
			if minx<=maxx and miny<=maxy:self.occ=scorer.occ_mark_clear(self.occ,minx,miny,maxx-minx+1,maxy-miny+1)
			self.bld_counts[bn]=max(0,self.bld_counts.get(bn,0)-1);self._update_cov_feats_global();self._emit_change('demolished',bn,minx,miny,maxx-minx+1,maxy-miny+1);_refresh_stats(self.board);return True
		else:
			bn=str(v);del self.board[pos];_log_demolido(bn,'1x1');self.occ=scorer.occ_mark_clear(self.occ,x,y,1,1)
			if bn in SERVICIOS:self._srv_apply_zone(bn,x,y,-1)
			self.bld_counts[bn]=max(0,self.bld_counts.get(bn,0)-1);self._update_cov_feats_global();self._emit_change('demolished',bn,x,y,1,1);_refresh_stats(self.board)
		return True
	def _shape_action_reward(self,bn,x,y,ok,prev_s,post_s):
		des0,mix0,neg0=_ec_terms(prev_s);des1,mix1,neg1=_ec_terms(post_s);r=.0;r+=K_RES_POS*(neg0-neg1)+K_RES_NEG*neg1;r+=K_DESBAL*(des0-des1);r+=K_MIX*(mix0-mix1)
		try:r+=K_ECO_DELTA*(float(post_s.ecologiaTotal)-float(prev_s.ecologiaTotal))
		except Exception:pass
		try:r+=K_FEL_DELTA*(float(post_s.felicidadTotal)-float(prev_s.felicidadTotal))
		except Exception:pass
		def _r_suelo():
			nonlocal r;r+=K_SUELO_ADJ*_contar_suelos_alrededor_celda(self.board,(x,y))
			if _es_borde(x,y):r+=K_SOIL_BORDER
			r+=_surplus_penalty(post_s);return r
		def _r_demoler():nonlocal r;r+=_surplus_penalty(post_s);return r
		def _r_other():
			nonlocal r
			if ok and bn!='suelo':
				w_,h_=_tam(bn);coords=_bloque(x,y,w_,h_);needs_soil=False
				try:needs_soil=bool(R.requiere_suelo(bn))
				except Exception:needs_soil=False
				if coords and needs_soil:extra=max(0,_contar_suelos_ady_board(self.board,coords)-1);r+=K_EDIF_SOIL_EXTRA*extra
				try:
					if coords and not R.es_activo(self.board,bn,coords):r+=K_INACTIVE
				except Exception:pass
			r+=_surplus_penalty(post_s);return r
		if bn=='suelo'and ok:return _r_suelo()
		return{'demoler':_r_demoler}.get(bn,_r_other)()
	def _shape_servicios_gain(self,bn,x,y,ok):
		if not ok or bn not in SERVICIOS:return .0
		zona=self._zona(bn,(x,y))
		if not zona:return .0
		L=len(SERVICIOS);k=SERV_IDX[bn];ref=self.srv_ref;overlap_same=sum(1 for(cx,cy)in zona if ref[(cy*S+cx)*L+k]>0);gain=len(zona)-overlap_same;frac_gain=gain/len(zona)if len(zona)else .0;ofrac_same=overlap_same/len(zona)if len(zona)else .0;inside=1 if ref[(y*S+x)*L+k]>0 else 0
		def any_any_service(px,py):i=(py*S+px)*L;return ref[i]>0 or ref[i+1]>0 or ref[i+2]>0 or ref[i+3]>0
		overlap_any=sum(1 for(cx,cy)in zona if any_any_service(cx,cy));ofrac_any=overlap_any/len(zona)if len(zona)else .0;origin_hits=0
		for stype in SERVICIOS:
			kk=SERV_IDX[stype]
			if ref[(y*S+x)*L+kk]>0:origin_hits+=1
		excess=max(0,origin_hits-SRV_OVERLAP_FREE_ORIGINS);reward=.0;reward+=K_SRV_GAIN*frac_gain;reward+=K_SRV_OVERLAP*ofrac_same;reward+=K_SRV_OVERLAP_SOFT*ofrac_any;reward+=K_SRV_INSIDE*float(inside)
		if excess>0:reward+=K_SRV_OVERLAP_HARD*float(excess)*ofrac_any
		orig=self.srv_origins.get(bn,[])
		if orig:mind=min(abs(ox-x)+abs(oy-y)for(ox,oy)in orig)if orig else 999;prox_pen=max(.0,(1e1-float(mind))/1e1);reward+=-.6*prox_pen
		return float(reward)
	@staticmethod
	@jax.jit
	def _returns(r,gamma=.98):
		def scan_fn(G,rr):G=rr+gamma*G;return G,G
		G,out=jax.lax.scan(scan_fn,jnp.float32(.0),r[::-1]);return out[::-1]
	def _save_ckpt(self):
		try:os.makedirs(os.path.dirname(POLICY_BIN),exist_ok=True);phi64=jnp.asarray(self._phi,jnp.float64);_save_policy_bin(self.agent,phi64,POLICY_BIN,version=1)
		except Exception as e:
			try:print('[AUTO] fallo al guardar .bin:',repr(e))
			except Exception:pass
	def save_now(self):
		try:
			self.running=False
			if getattr(self,'thread',None):self.thread.join(timeout=1.)
		except Exception:pass
		try:
			if getattr(self,'writer',None):
				for fn in(self.writer.flush,self.writer.sync,self.writer.close):
					try:fn()
					except Exception:pass
		except Exception:pass
		self._save_ckpt()
	def _map_full(self):return len(self.board)>=S*S*.98
	def _intrinsic(self,next_obs):
		if not self.rnd:return .0
		ri=.0
		try:
			if hasattr(self.rnd,'compute_and_update'):ri=float(self.rnd.compute_and_update(next_obs))
			else:
				ri=float(self.rnd.compute(next_obs))
				try:self.rnd.update(next_obs)
				except Exception:pass
		except Exception:ri=.0
		return self.rnd_scale*ri
	def _exec_demoler(self,bn,u,v,s0,pre_cov_stats):
		pre_ov=self._total_overlap_ratio()
		if self._demo_target is not None:x,y=self._demo_target;self._demo_target=None
		else:
			try:
				target=self._pick_redundant_service_origin()
				if target is not None:x,y=target
				else:x=y=None
			except Exception:x=y=None
		if x is None or y is None:
			if self.lotes_on:x,y=self._rand_in_unlocked()
			else:x=int(u*(S-1));y=int(v*(S-1))
		ok=self._demoler_en((x,y));self._update_cov_feats_global();obs1,s1=_extract_features(self.board);post_ov=self._total_overlap_ratio();r=K_DEMOLER_BASE+self._shape_action_reward(bn,x,y,ok,s0,s1);r+=K_DEMO_OVERLAP_GAIN*max(.0,pre_ov-post_ov);r+=self._intrinsic(obs1);return ok,r,obs1,s1
	def _exec_place(self,bn,u,v,s0,pre_cov_stats):
		if bn=='decoracion'and getattr(s0,'ecologiaTotal',0)>=100:return False,K_FAIL,self.last_obs,s0
		if bn in PROV_TO_RES:
			rname=PROV_TO_RES[bn];bal=getattr(s0,rname+'Total')+getattr(s0,rname+'Usada')
			if bal>=0 and self._provider_exists(bn):return False,K_FAIL,self.last_obs,s0
		if self.lotes_on:x,y=self._rand_in_unlocked()
		else:w,h=_tam(bn);x=int(u*max(1,S-w));y=int(v*max(1,S-h))
		try:cand=scorer.best_spot(self.board,bn,{k:_tam(k)for k in ED},topk=512,occ=self.occ)
		except Exception:cand=None
		if cand and self._is_allowed(cand[0],cand[1]):x,y=cand
		srv_bonus=self._shape_servicios_gain(bn,x,y,True);pre_ok=self._ensure_prereqs(bn,x,y)
		try:print(f"[AUTO] intentando {bn} en x={x} y={y} (S={S}) pre_ok={pre_ok}")
		except Exception:pass
		ok=self._try_place(bn,x,y)if pre_ok else False;self._update_cov_feats_global();obs1,s1=_extract_features(self.board);r=self._shape_action_reward(bn,x,y,ok,s0,s1)+(srv_bonus if ok else .0);r+=self._shape_bld_coverage(pre_cov_stats,self._coverage_stats(),bn,x,y,ok)
		if not ok:r+=K_FAIL
		r+=self._intrinsic(obs1);return ok,r,obs1,s1
	def _step_maint_idle(self,obs,s0,pre_cov_stats,priority_bn=None):obs1,s1=_extract_features(self.board);a_type=IDX['suelo'];u=self._rand();v=self._rand();bn='suelo';r=self._intrinsic(obs1);ok=True;return ok,r,obs1,s1,a_type,u,v,bn
	def _step_maint_build(self,obs,s0,pre_cov_stats,priority_bn=None):
		a_type,u,v,_probs=self.agent.act(obs);bn=priority_bn if priority_bn in IDX else ACTION_SET[a_type]
		if bn in IDX:a_type=IDX[bn]
		if bn=='decoracion'and getattr(s0,'ecologiaTotal',0)>=100:obs1,s1=_extract_features(self.board);ok=False;r=K_FAIL+self._intrinsic(obs1);return ok,r,obs1,s1,a_type,u,v,bn
		if bn in PROV_TO_RES:
			rname=PROV_TO_RES[bn];bal=getattr(s0,rname+'Total')+getattr(s0,rname+'Usada')
			if bal>=0 and self._provider_exists(bn):obs1,s1=_extract_features(self.board);ok=False;r=K_FAIL+self._intrinsic(obs1);return ok,r,obs1,s1,a_type,u,v,bn
		if self.lotes_on:x,y=self._rand_in_unlocked()
		else:x=int(u*S);y=int(v*S)
		self._clear_for_place(bn,x,y);pre_ok=self._ensure_prereqs(bn,x,y);ok=self._place(bn,x,y)if pre_ok else False;self._update_cov_feats_global();obs1,s1=_extract_features(self.board);srv_bonus=self._shape_servicios_gain(bn,x,y,ok);r=self._shape_action_reward(bn,x,y,ok,s0,s1)+(srv_bonus if ok else .0);r+=self._intrinsic(obs1);r+=self._shape_bld_coverage(pre_cov_stats,self._coverage_stats(),bn,x,y,ok)
		if not ok:r+=K_FAIL
		return ok,r,obs1,s1,a_type,u,v,bn
	def _step_force_demo(self,obs,s0,pre_cov_stats,priority_bn=None):a_type=IDX['demoler'];u=self._rand();v=self._rand();bn='demoler';ok,r,obs1,s1=self._exec_demoler(bn,u,v,s0,pre_cov_stats);return ok,r,obs1,s1,a_type,u,v,bn
	def _step_act(self,obs,s0,pre_cov_stats,priority_bn=None):
		a_type,u,v,probs=self.agent.act(obs);bn=ACTION_SET[a_type]
		try:priors_cov={k:self.srv_cov_counts[SERV_IDX[k]]/float(S*S)for k in SERVICIOS};p2=R.reweight_probs(probs,list(ACTION_SET),lock_resources=self.last_lock_resources,servicio_coverage=priors_cov)
		except Exception:p2=probs
		p2=[max(.0,float(x))if x==x else .0 for x in p2]
		for res in self.last_lock_resources:
			prov=DEFICIT_TO_BN.get(res)
			if prov in IDX and self._provider_exists(prov):p2[IDX[prov]]=.0
		if len(self.board)<3:p2[IDX['demoler']]=.0;p2[IDX['suelo']]=p2[IDX['suelo']]*1.5+.1
		for res in self.last_need_resources:
			prov=DEFICIT_TO_BN.get(res)
			if prov in IDX:p2[IDX[prov]]=p2[IDX[prov]]*2.+.05
		has_suelo=any(v2=='suelo'for v2 in self.board.values());has_decor=any(v2=='decoracion'for v2 in self.board.values())
		if not has_suelo:
			for a in ACTION_SET:
				if a in IDX and R.requiere_suelo(a):p2[IDX[a]]*=.35
			p2[IDX['suelo']]=p2[IDX['suelo']]*1.7+.15
		if not has_decor:
			for a in ACTION_SET:
				if a in IDX and R.requiere_decoracion(a):p2[IDX[a]]*=.35
			p2[IDX['decoracion']]=p2[IDX['decoracion']]*1.7+.15
		cand=self._neg_env_targets()
		if cand:target_decor=cand[0][0];top=cand[0][1];boost=min(DECOR_BOOST,1.+top/1e1);p2[IDX['decoracion']]=p2[IDX['decoracion']]*boost+.25;p2[IDX['demoler']]*=.5
		else:target_decor=None
		if getattr(self.last_stats,'ecologiaTotal',0)>=100:p2[IDX['decoracion']]=.0
		try:demo_target=self._pick_redundant_service_origin()
		except Exception:demo_target=None
		self._demo_target=demo_target;dup_boost=.0
		if demo_target is not None:dup_boost+=.6
		if self.same_fail_count>=8:dup_boost+=.35
		if self.steps-max(self.last_change_step,self.last_progress_step)>STUCK_STEPS//2:dup_boost+=.4
		p2[IDX['demoler']]=p2[IDX['demoler']]*.2+dup_boost+.05
		if sum(p2)<=.0:
			p2=[1.]*NA
			if getattr(self.last_stats,'ecologiaTotal',0)>=100:p2[IDX['decoracion']]=.0
		cd_exp=self.cooldown.get(a_type,-1);a_type=self._resample_action(p2,a_type)if cd_exp>=self.steps else self._resample_action(p2,None);bn=ACTION_SET[a_type]
		if self.cooldown.get(a_type,-1)>=self.steps and self.last_need_resources:
			want=self._choice(list(self.last_need_resources));forced=DEFICIT_TO_BN.get(want)if want else None
			if forced in IDX:a_type=IDX[forced];bn=forced
		if len(self.board)==0 and bn=='demoler':bn='suelo';a_type=IDX['suelo'];u=self._rand();v=self._rand()
		if bn=='decoracion'and target_decor:
			tx,ty=target_decor
			if self._is_allowed(tx,ty):u=tx/max(1,S-1);v=ty/max(1,S-1)
		if self.steps%50==0:
			try:print(f"[AUTO] step={self.steps} acción={bn}{ u=:.3f}{ v=:.3f} locks={sorted(self.last_lock_resources)} board_len={len(self.board)}")
			except Exception:pass
		ok,r,obs1,s1={True:self._exec_demoler,False:self._exec_place}[bn=='demoler'](bn,u,v,s0,pre_cov_stats);return ok,r,obs1,s1,a_type,u,v,bn
	def simulation_step(self):
		with self.lock:
			if self.game_over:return False
			if self._srv_dirty and self.steps%self.cache_period==0:self._rebuild_srv_cache_full();self._srv_dirty=False;self._update_cov_feats_global()
			stuck=self.steps-max(self.last_change_step,self.last_progress_step)>STUCK_STEPS;force_demo=stuck and self.fails>200;global _LOCKS,_GLOBAL_STEPS;_LOCKS=set(self.last_lock_resources);self._update_cov_feats_global();obs=self.last_obs;s0=self.last_stats;_GLOBAL_STEPS=self.steps;is_full=self._map_full()and MAINTENANCE_ON_FULL;pre_cov_stats=self._coverage_stats();self._enter_exit_maintenance(is_full);step_key='act';priority_bn=None
			if is_full:need,priority_bn=self._needs_improvement();step_key='maint_idle'if not need and MAINT_IDLE_OK else'maint_build'
			else:step_key='force_demo'if force_demo else'act'
			ok,r,obs1,s1,a_type,u,v,bn=self._step_handlers[step_key](obs,s0,pre_cov_stats,priority_bn)
			try:print(f"[AUTO] post-acción bn={bn} ok={ok} fails={self.fails} celdas={len(self.board)}")
			except Exception:pass
			sig1=_sig_progress(s1)
			if sig1!=self.progress_sig:self.progress_sig=sig1;self.last_progress_step=self.steps
			if ok:self.same_fail_type=None;self.same_fail_count=0
			else:
				if bn==self.same_fail_type:self.same_fail_count+=1
				else:self.same_fail_type=bn;self.same_fail_count=1
				if self.same_fail_count>=MAX_SAME_FAILS:self.cooldown[IDX[bn]]=self.steps+COOLDOWN_STEPS;self.same_fail_type=None;self.same_fail_count=0
			self.last_obs,self.last_stats=obs1,s1;self.last_terms=_ec_terms(s1);self.last_lock_resources=_lock_from_stats(s1);self.last_need_resources=_needs_from_stats(s1);self.updated=True;self.steps+=1;self.fails=0 if ok else self.fails+1;self.buf_o.append(obs);self.buf_a.append(a_type);self.buf_u.append(u);self.buf_v.append(v);self.buf_r.append(float(r))
			try:
				if self.writer:self.writer.append(int(ok),a_type,float(u),float(v),float(r),float(sum(self.last_terms)),obs)
			except Exception:pass
			self._maybe_unlock_next(s1)
			if len(self.buf_r)>=64:
				r_vec=jnp.array(self.buf_r,dtype=jnp.float32);g_vec=self._returns(r_vec);self.baseline=.9*self.baseline+.1*float(jnp.mean(g_vec));adv=g_vec=self.baseline
				try:self.agent.update(jnp.stack(self.buf_o),jnp.array(self.buf_a),jnp.array(self.buf_u),jnp.array(self.buf_v),adv,lr=.0003)
				except Exception:pass
				self.buf_o.clear();self.buf_a.clear();self.buf_u.clear();self.buf_v.clear();self.buf_r.clear()
		if self.steps%STEPWRITER_FLUSH_EVERY==0:
			try:
				if self.writer:self.writer.flush()
			except Exception:pass
		if self.steps%AUTOSYNC_EVERY==0:
			try:
				if self.writer:self.writer.sync()
			except Exception:pass
		if self.steps%AUTOSAVE_EVERY==0:self._save_ckpt()
		if self.fails>5000:
			print('Juego terminado (demasiados fallos consecutivos)');self.game_over=True;self.running=False
			try:
				if self.writer:self.writer.close()
			except Exception:pass
			self._save_ckpt()
		return True
	def _enter_exit_maintenance(self,is_full):
		def _enter():
			try:print('[AUTO] Entrando en MODO MANTENIMIENTO (tablero lleno).')
			except Exception:pass
		def _exit():
			try:print('[AUTO] Saliendo de MODO MANTENIMIENTO.')
			except Exception:pass
		{(False,True):_enter,(True,False):_exit}.get((self._in_maintenance,is_full),_noop)();self._in_maintenance=is_full
	def _run(self):
		print('[AUTO] hilo iniciado')
		try:
			while self.running and not self.game_over:
				try:self.simulation_step()
				except Exception as e:print('[AUTO] excepción en step:',repr(e))
				time.sleep(self.tick_sleep)
		finally:print(f"[AUTO] hilo finalizado (running={self.running}, game_over={self.game_over})")
	def actualizar(self,simulation_steps,simulation_speed):
		if simulation_steps and not self.running and not self.game_over:self.running=True;self.thread=threading.Thread(target=self._run,daemon=False);self.thread.start()
		elif not simulation_steps and self.running:self.running=False
		return simulation_speed+simulation_steps
