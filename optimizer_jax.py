# optimizer_jax.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,Tuple,List,Optional,Set,Callable
import random,time,jax.numpy as jnp
from jax import jit
from functools import partial
from config import BOARD_SIZE
import edificios,scorer_jax,restricciones
S=int(BOARD_SIZE)
@dataclass(frozen=True)
class Move:op:str;edificio:str;x:int;y:int
def _is_origin_map(board):
	origins={};result={}
	for((x,y),val)in board.items():
		if isinstance(val,tuple):bn,raw_iid=val;iid=str(raw_iid);prev=origins.get(iid);origins[iid]=(x,y)if prev is None else(min(prev[0],x),min(prev[1],y))
	def _reg_tuple(x,y,val):bn,raw_iid=val;iid=str(raw_iid);origins.get(iid)==(x,y)and result.__setitem__((x,y),(bn,iid))
	def _reg_str(x,y,val):val and result.__setitem__((x,y),(val,None))
	for((x,y),val)in board.items():{tuple:_reg_tuple,str:_reg_str}.get(type(val),lambda*_:None)(x,y,val)
	return result
def _bloque_libre(board,x,y,w,h):return not any((x+i,y+j)in board for i in range(w)for j in range(h))
def _sumas_base(board):
	E=edificios.edificios;origenes=_is_origin_map(board);total_emp=0;total_res=0;emp_ind=0;emp_com=0;used=dict(energia=0,agua=0,basura=0,comida=0);supply=dict(energia=0,agua=0,basura=0,comida=0);cap_prov={'refineria':10000,'agua':40000,'lecheria':4000,'depuradora':40000};inactivos=0;cant_suelo=0;sup_acc={'refineria':lambda:supply.__setitem__('energia',supply['energia']+cap_prov['refineria']),'agua':lambda:supply.__setitem__('agua',supply['agua']+cap_prov['agua']),'lecheria':lambda:supply.__setitem__('comida',supply['comida']+cap_prov['lecheria']),'depuradora':lambda:supply.__setitem__('basura',supply['basura']+cap_prov['depuradora'])}
	for((ox,oy),(bn,_iid))in origenes.items():
		def _acum_suelo():nonlocal cant_suelo;cant_suelo+=1
		def _acum_edif():
			nonlocal total_emp,total_res,emp_ind,emp_com,inactivos
			try:a=E[bn]
			except KeyError:return
			d=a.to_dict;emp=d['empleos'];res=d['residentes'];total_emp+=emp;total_res+=res;tip=str(getattr(a.tipo,'value',a.tipo));inc_ind,inc_com={'industria':(emp,0),'comercio':(0,emp)}.get(tip,(0,0));emp_ind+=inc_ind;emp_com+=inc_com;used['energia']+=d['energia'];used['agua']+=d['agua'];used['basura']+=d['basura'];used['comida']+=d['comida'];sup_acc.get(bn,lambda:None)();inactivos+=int(not restricciones.es_activo(bn,(ox,oy),board))
		{True:_acum_suelo,False:_acum_edif}[bn=='suelo']()
	return dict(total_emp=total_emp,total_res=total_res,emp_ind=emp_ind,emp_com=emp_com,used=used,supply=supply,inactivos=inactivos,cant_suelo=cant_suelo)
@partial(jit,static_argnums=())
def _score_batch_jax(base_emp,base_res,base_emp_ind,base_emp_com,base_used_e,base_used_a,base_used_b,base_used_c,base_sup_e,base_sup_a,base_sup_b,base_sup_c,base_inact,base_suelo,de_emp,de_res,de_emp_ind,de_emp_com,de_used_e,de_used_a,de_used_b,de_used_c,de_sup_e,de_sup_a,de_sup_b,de_sup_c,de_inact,de_suelo):
	emp=base_emp+de_emp;res=base_res+de_res;ind=base_emp_ind+de_emp_ind;com=base_emp_com+de_emp_com;ue=base_used_e+de_used_e;ua=base_used_a+de_used_a;ub=base_used_b+de_used_b;uc=base_used_c+de_used_c;se=base_sup_e+de_sup_e;sa=base_sup_a+de_sup_a;sb=base_sup_b+de_sup_b;sc=base_sup_c+de_sup_c;ina=base_inact+de_inact;suelo=base_suelo+de_suelo;pen_desemp=jnp.abs(emp-res).astype(jnp.float32);te=jnp.maximum(emp,1);p_ind=ind/te;p_com=com/te;pen_ratio=(p_ind-1./3.)**2+(p_com-2./3.)**2
	def res_pen(used,sup):deficit=jnp.maximum(.0,used-sup);surplus=jnp.maximum(.0,sup-used);return 5e1*deficit+.02*surplus
	pen_rec=res_pen(ue,se)+res_pen(ua,sa)+res_pen(ub,sb)+res_pen(uc,sc);pen_inact=1e3*ina.astype(jnp.float32);pen_suelo=.2*suelo.astype(jnp.float32);score=1e6-(2e1*pen_desemp+2e3*pen_ratio+pen_rec+pen_inact+pen_suelo);return score
def _proponer_candidatos(board,batch,rng,usar_best_spot=True):
	E=edificios.edificios;nombres=[k for k in E.keys()if k and k!='suelo'];sizes={k:tuple(map(int,E[k].tamanio))for k in nombres};candidatos=[];intentos_random_por_fallo=8
	def _rand_spot(bn,w,h):coords=((rng.randrange(0,S-w+1),rng.randrange(0,S-h+1))for _ in range(intentos_random_por_fallo));return next(((x,y)for(x,y)in coords if _bloque_libre(board,x,y,w,h)),None)
	for _ in range(batch):bn=rng.choice(nombres);w,h=sizes[bn];f_best=lambda:scorer_jax.best_spot(board,bn,sizes);f_rand=lambda:_rand_spot(bn,w,h);pos=next((p for p in(f_best(),f_rand())if p is not None),None);pos and candidatos.append((bn,pos[0],pos[1]))
	return candidatos
def _delta_para(bn):
	try:E=edificios.edificios[bn]
	except KeyError:return 0,0,0,0,0,0,0,0,0,0,0,0
	d=E.to_dict;emp=int(d['empleos']);res=int(d['residentes']);tip=str(getattr(E.tipo,'value',E.tipo));emp_ind,emp_com={'industria':(emp,0),'comercio':(0,emp)}.get(tip,(0,0));ue,ua,ub,uc=int(d['energia']),int(d['agua']),int(d['basura']),int(d['comida']);supply_map={'refineria':(10000,0,0,0),'agua':(0,40000,0,0),'depuradora':(0,0,40000,0),'lecheria':(0,0,0,4000)};se,sa,sb,sc=supply_map.get(bn,(0,0,0,0));return emp,res,emp_ind,emp_com,ue,ua,ub,uc,se,sa,sb,sc
def _aplicar_move(board,mv):
	E=edificios.edificios;w,h=map(int,E[mv.edificio].tamanio);iid=str(time.time_ns())
	for i in range(w):
		for j in range(h):board[mv.x+i,mv.y+j]=mv.edificio,iid
def mejorar(board,pasos=200,batch=64,seed=None):
	rng=random.Random(seed or time.time_ns()&4294967295);moves=[];base=_sumas_base(board);base_emp=int(base['total_emp']);base_res=int(base['total_res']);base_emp_ind=int(base['emp_ind']);base_emp_com=int(base['emp_com']);base_used_e=float(base['used']['energia']);base_used_a=float(base['used']['agua']);base_used_b=float(base['used']['basura']);base_used_c=float(base['used']['comida']);base_sup_e=float(base['supply']['energia']);base_sup_a=float(base['supply']['agua']);base_sup_b=float(base['supply']['basura']);base_sup_c=float(base['supply']['comida']);base_inact=int(base['inactivos']);base_suelo=int(base['cant_suelo']);curr_score=_score_batch_jax(base_emp,base_res,base_emp_ind,base_emp_com,base_used_e,base_used_a,base_used_b,base_used_c,base_sup_e,base_sup_a,base_sup_b,base_sup_c,base_inact,base_suelo,jnp.array([0]),jnp.array([0]),jnp.array([0]),jnp.array([0]),jnp.array([.0]),jnp.array([.0]),jnp.array([.0]),jnp.array([.0]),jnp.array([.0]),jnp.array([.0]),jnp.array([.0]),jnp.array([.0]),jnp.array([0]),jnp.array([0]))[0];tabu=set();E=edificios.edificios;sizes={k:tuple(map(int,E[k].tamanio))for k in E.keys()if k}
	def _add_tabu_from(seq):
		k=max(1,batch//4)
		for t in seq[:k]:tabu.add(t)
	for _step in range(pasos):
		cands=_proponer_candidatos(board,batch,rng,usar_best_spot=True)
		if not cands:break
		cand_iter=filter(lambda t:t not in tabu,cands);cand_iter=filter(lambda t:_bloque_libre(board,t[1],t[2],*sizes[t[0]]),cand_iter);cand_iter=filter(lambda t:restricciones.puede_colocar(t[0],(t[1],t[2]),board),cand_iter);deltas=[];metas=[];de_inact_list=[];de_suelo_list=[]
		for(bn,x,y)in cand_iter:de_emp,de_res,de_emp_ind,de_emp_com,de_ue,de_ua,de_ub,de_uc,de_se,de_sa,de_sb,de_sc=_delta_para(bn);deltas.append((de_emp,de_res,de_emp_ind,de_emp_com,float(de_ue),float(de_ua),float(de_ub),float(de_uc),float(de_se),float(de_sa),float(de_sb),float(de_sc)));metas.append((bn,x,y));de_inact_list.append(int(not restricciones.es_activo(bn,(x,y),board)));de_suelo_list.append(int(bn=='suelo'))
		not deltas and _add_tabu_from(cands)and None
		if not deltas:continue
		de_emp=jnp.array([d[0]for d in deltas]);de_res=jnp.array([d[1]for d in deltas]);de_emp_ind=jnp.array([d[2]for d in deltas]);de_emp_com=jnp.array([d[3]for d in deltas]);de_used_e=jnp.array([d[4]for d in deltas],dtype=jnp.float32);de_used_a=jnp.array([d[5]for d in deltas],dtype=jnp.float32);de_used_b=jnp.array([d[6]for d in deltas],dtype=jnp.float32);de_used_c=jnp.array([d[7]for d in deltas],dtype=jnp.float32);de_sup_e=jnp.array([d[8]for d in deltas],dtype=jnp.float32);de_sup_a=jnp.array([d[9]for d in deltas],dtype=jnp.float32);de_sup_b=jnp.array([d[10]for d in deltas],dtype=jnp.float32);de_sup_c=jnp.array([d[11]for d in deltas],dtype=jnp.float32);de_inact=jnp.array(de_inact_list);de_suelo=jnp.array(de_suelo_list);scores=_score_batch_jax(base_emp,base_res,base_emp_ind,base_emp_com,base_used_e,base_used_a,base_used_b,base_used_c,base_sup_e,base_sup_a,base_sup_b,base_sup_c,base_inact,base_suelo,de_emp,de_res,de_emp_ind,de_emp_com,de_used_e,de_used_a,de_used_b,de_used_c,de_sup_e,de_sup_a,de_sup_b,de_sup_c,de_inact,de_suelo);best_idx=int(jnp.argmax(scores));best_score=float(scores[best_idx])
		if best_score<=float(curr_score):_add_tabu_from(metas);continue
		bn,x,y=metas[best_idx];mv=Move('add',bn,x,y);_aplicar_move(board,mv);moves.append(mv);tabu.add((bn,x,y));d=deltas[best_idx];base_emp+=d[0];base_res+=d[1];base_emp_ind+=d[2];base_emp_com+=d[3];base_used_e+=d[4];base_used_a+=d[5];base_used_b+=d[6];base_used_c+=d[7];base_sup_e+=d[8];base_sup_a+=d[9];base_sup_b+=d[10];base_sup_c+=d[11];base_inact+=int(de_inact_list[best_idx]);base_suelo+=int(de_suelo_list[best_idx]);curr_score=best_score
	return moves