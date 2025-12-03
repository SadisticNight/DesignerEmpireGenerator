# ml_policy.py
from jax import config as _jax_config
_jax_config.update('jax_enable_x64',True)
import os,struct,functools,jax,jax.numpy as jnp,optax
try:import rlax;_HAS_RLAX=True
except Exception:_HAS_RLAX=False
import capnp
_policy_schema=capnp.load('policy.capnp')
try:import xxhash;_HAS_XXH=True
except Exception:_HAS_XXH=False
def _he_init(key,shape_in,shape_out,scale=2.):return jax.random.normal(key,(shape_in,shape_out),dtype=jnp.float32)*jnp.sqrt(jnp.float32(scale/max(1,shape_in)))
def _init_params(key,din,dh,na):k1,k2,k3=jax.random.split(key,3);W1=_he_init(k1,din,dh);b1=jnp.zeros((dh,),jnp.float32);Wt=_he_init(k2,dh,na);bt=jnp.zeros((na,),jnp.float32);Wp=_he_init(k3,dh,4);bp=jnp.zeros((4,),jnp.float32);return{'W1':W1,'b1':b1,'Wt':Wt,'bt':bt,'Wp':Wp,'bp':bp}
@jax.jit
def _forward(params,x):h=jnp.tanh(x@params['W1']+params['b1']);logits=h@params['Wt']+params['bt'];head_uv=h@params['Wp']+params['bp'];return h,logits,head_uv
@jax.jit
def _forward_ctx(params,x,AC):h=jnp.tanh(x@params['W1']+params['b1']);logits=AC@params['Pa']@h;head_uv=h@params['Wp']+params['bp'];return h,logits,head_uv
_TWO_PI=jnp.float32(6.283185307179586)
_LOG2=jnp.float32(.6931471805599453)
def _logN(z,mu,logstd):std=jnp.exp(logstd);zscore=(z-mu)/std;return jnp.sum(-.5*zscore**2-logstd-.5*jnp.log(_TWO_PI))
def _squashed_gauss_logprob_uv(mu,logstd,target_u01):eps=jnp.float32(1e-06);y=jnp.clip(target_u01*2.-1.,-1.+eps,1.-eps);z=jnp.arctanh(y);logp_z=_logN(z,mu,logstd);corr=jnp.sum(_LOG2-jnp.log1p(-y*y));return logp_z+corr
def _gaussian_entropy_2d(logstd_xy):return jnp.sum(logstd_xy+.5*(jnp.log(_TWO_PI)+1.))
def _loss_pg_noctx(params,X,A,U,V,ADV,entropy_coef):
	def per_sample(x,a,u,v,adv):_,logits,head=_forward(params,x);logp_type=jax.nn.log_softmax(logits)[a];mu=head[0:2];logstd=jnp.clip(head[2:4],-3.,1.5);logp_uv=_squashed_gauss_logprob_uv(mu,logstd,jnp.stack([u,v]));ent_cat=-jnp.sum(jax.nn.softmax(logits)*jax.nn.log_softmax(logits));ent_uv=_gaussian_entropy_2d(logstd);return-(logp_type+logp_uv)*adv-entropy_coef*(ent_cat+ent_uv)
	return jnp.mean(jax.vmap(per_sample)(X,A,U,V,ADV))
def _loss_pg_ctx(params,X,A,U,V,ADV,entropy_coef,AC):
	def per_sample(x,a,u,v,adv):_,logits,head=_forward_ctx(params,x,AC);logp_type=jax.nn.log_softmax(logits)[a];mu=head[0:2];logstd=jnp.clip(head[2:4],-3.,1.5);logp_uv=_squashed_gauss_logprob_uv(mu,logstd,jnp.stack([u,v]));ent_cat=-jnp.sum(jax.nn.softmax(logits)*jax.nn.log_softmax(logits));ent_uv=_gaussian_entropy_2d(logstd);return-(logp_type+logp_uv)*adv-entropy_coef*(ent_cat+ent_uv)
	return jnp.mean(jax.vmap(per_sample)(X,A,U,V,ADV))
def _loss_bc_noctx(params,X,A,U,V,coord_weight):
	def per_sample(x,a,u,v):_,logits,head=_forward(params,x);ce=-jax.nn.log_softmax(logits)[a];mu01=jnp.clip((jnp.tanh(head[0:2])+1.)/2.,.0,1.);mse=jnp.mean((mu01-jnp.stack([u,v]))**2);return ce+coord_weight*mse
	return jnp.mean(jax.vmap(per_sample)(X,A,U,V))
def _loss_bc_ctx(params,X,A,U,V,coord_weight,AC):
	def per_sample(x,a,u,v):_,logits,head=_forward_ctx(params,x,AC);ce=-jax.nn.log_softmax(logits)[a];mu01=jnp.clip((jnp.tanh(head[0:2])+1.)/2.,.0,1.);mse=jnp.mean((mu01-jnp.stack([u,v]))**2);return ce+coord_weight*mse
	return jnp.mean(jax.vmap(per_sample)(X,A,U,V))
def _ac_id_from_AC(AC):
	if not _HAS_XXH or AC is None:return
	arr=jnp.asarray(AC,jnp.float64);rows,cols=int(arr.shape[0]),int(arr.shape[1]);h=xxhash.xxh3_64();h.update(struct.pack('<Q',rows));h.update(struct.pack('<Q',cols));flat=jnp.ravel(arr);host=jax.device_get(flat);h.update(host.tobytes(order='C'));return h.intdigest()
@jax.jit
def _discounted_returns_jax(r,gamma):
	def scan(G,rt):G=rt+gamma*G;return G,G
	_,out=jax.lax.scan(scan,jnp.float32(.0),r[::-1]);return out[::-1]
@jax.jit
def _lambda_returns_pg_no_value(r,gamma,lam):eff=jnp.float32(gamma)*jnp.float32(lam);return _discounted_returns_jax(r,eff)
class Policy:
	__slots__='key','params','na','entropy_coef','action_context','opt','opt_state','clip_norm','weight_decay'
	def __init__(self,action_count,obs_dim,seed=123,hidden=128,entropy_coef=.001,clip_norm=1.,weight_decay=.0):self.key=jax.random.PRNGKey(seed);self.na=int(action_count);self.entropy_coef=float(entropy_coef);self.clip_norm=float(clip_norm);self.weight_decay=float(weight_decay);self.params=_init_params(self.key,int(obs_dim),int(hidden),int(action_count));self.action_context=None;self.opt=optax.chain(optax.clip_by_global_norm(self.clip_norm),optax.adamw(learning_rate=1.,weight_decay=self.weight_decay));self.opt_state=self.opt.init(self.params)
	def _ensure_ctx(self):
		AC=self.action_context
		if AC is None:return
		AC=jnp.asarray(AC,jnp.float32)
		if AC.shape[0]!=self.na:AC=AC[:self.na]if AC.shape[0]>self.na else jnp.pad(AC,((0,self.na-AC.shape[0]),(0,0)))
		self.action_context=AC;dh=int(self.params['W1'].shape[1]);K=int(AC.shape[1])
		if'Pa'not in self.params or tuple(self.params['Pa'].shape)!=(K,dh):self.key,k=jax.random.split(self.key,2);Pa=_he_init(k,K,dh);self.params=dict(self.params,Pa=Pa);self.opt_state=self.opt.init(self.params)
		return AC
	def act(self,obs):
		x=jnp.asarray(obs,jnp.float32);AC=self._ensure_ctx();self.key,sk1,sk2=jax.random.split(self.key,3)
		if AC is not None:_,logits,head=_forward_ctx(self.params,x,AC)
		else:_,logits,head=_forward(self.params,x)
		p=jax.nn.softmax(logits);a_type=int(jax.random.categorical(sk1,jnp.log(p)));mu=head[0:2];logstd=jnp.clip(head[2:4],-3.,1.5);std=jnp.exp(logstd);noise=jax.random.normal(sk2,(2,),dtype=jnp.float32);z=mu+noise*std;u01=jnp.clip((jnp.tanh(z)+1.)/2.,.0,1.);return a_type,float(u01[0]),float(u01[1]),p
	def update(self,X,A,U,V,ADV,lr=.0003):
		X=jnp.asarray(X,jnp.float32);A=jnp.asarray(A,jnp.int32);U=jnp.asarray(U,jnp.float32);V=jnp.asarray(V,jnp.float32);ADV=jnp.asarray(ADV,jnp.float32);AC=self._ensure_ctx()
		if AC is None:loss_fn=functools.partial(_loss_pg_noctx,entropy_coef=jnp.float32(self.entropy_coef));loss,grads=jax.value_and_grad(loss_fn)(self.params,X,A,U,V,ADV)
		else:loss_fn=functools.partial(_loss_pg_ctx,entropy_coef=jnp.float32(self.entropy_coef),AC=AC);loss,grads=jax.value_and_grad(loss_fn)(self.params,X,A,U,V,ADV)
		updates,self.opt_state=self.opt.update(grads,self.opt_state,params=self.params);updates=jax.tree_util.tree_map(lambda u:u*jnp.float32(lr),updates);self.params=optax.apply_updates(self.params,updates);return float(loss)
	def bc_update(self,X,A,U,V,lr=.0003,coord_weight=5.):
		X=jnp.asarray(X,jnp.float32);A=jnp.asarray(A,jnp.int32);U=jnp.asarray(U,jnp.float32);V=jnp.asarray(V,jnp.float32);AC=self._ensure_ctx()
		if AC is None:loss_fn=functools.partial(_loss_bc_noctx,coord_weight=jnp.float32(coord_weight));loss,grads=jax.value_and_grad(loss_fn)(self.params,X,A,U,V)
		else:loss_fn=functools.partial(_loss_bc_ctx,coord_weight=jnp.float32(coord_weight),AC=AC);loss,grads=jax.value_and_grad(loss_fn)(self.params,X,A,U,V)
		updates,self.opt_state=self.opt.update(grads,self.opt_state,params=self.params);updates=jax.tree_util.tree_map(lambda u:u*jnp.float32(lr),updates);self.params=optax.apply_updates(self.params,updates);return float(loss)
	def compute_returns(self,rewards,gamma=.98,lam=1.):
		r=jnp.asarray(rewards,jnp.float32)
		if _HAS_RLAX and lam>=.999:disc=jnp.full_like(r,jnp.float32(gamma));return rlax.discounted_returns(r,disc)
		if lam>=.999:return _discounted_returns_jax(r,jnp.float32(gamma))
		else:return _lambda_returns_pg_no_value(r,jnp.float32(gamma),jnp.float32(lam))
	def update_with_returns(self,X,A,U,V,rewards,gamma=.98,lam=1.,lr=.0003,baseline_m=.9,baseline=None):G=self.compute_returns(rewards,gamma,lam);b=jnp.mean(G)if baseline is None else jnp.float32(baseline);ADV=G-b;return self.update(X,A,U,V,ADV,lr=lr)
	def save(self,path):
		root,ext=os.path.splitext(path);bin_path=path if ext.lower()=='.bin'else root+'.bin';dirp=os.path.dirname(bin_path)
		if dirp:os.makedirs(dirp,exist_ok=True)
		msg=_policy_schema.PolicyCheckpoint.new_message();msg.version=1;msg.obsDim=int(self.params['W1'].shape[0]);msg.hidden=int(self.params['W1'].shape[1]);msg.actionCount=int(self.na);AC=self.action_context
		if AC is not None:ACf32=jnp.asarray(AC,jnp.float32);msg.phiRows=int(ACf32.shape[0]);msg.phiCols=int(ACf32.shape[1]);msg.acId=int(_ac_id_from_AC(ACf32)or 0)
		else:msg.phiRows=0;msg.phiCols=0;msg.acId=0
		def set_t2(dst,arr):a=jax.device_get(jnp.asarray(arr,jnp.float32));r,c=int(a.shape[0]),int(a.shape[1]);flat=a.reshape(-1).tolist();dst.rows=r;dst.cols=c;dst.init('data',len(flat));dst.data[:]=[float(x)for x in flat]
		def set_t1(dst,arr):a=jax.device_get(jnp.asarray(arr,jnp.float32)).reshape(-1).tolist();dst.init('data',len(a));dst.data[:]=[float(x)for x in a]
		set_t2(msg.w1,self.params['W1']);set_t1(msg.b1,self.params['b1']);set_t2(msg.wt,self.params['Wt']);set_t1(msg.bt,self.params['bt']);set_t2(msg.wp,self.params['Wp']);set_t1(msg.bp,self.params['bp']);has_pa='Pa'in self.params;msg.hasPa=bool(has_pa)
		if has_pa:set_t2(msg.pa,self.params['Pa'])
		tmp=bin_path+'.tmp'
		with open(tmp,'wb')as f:msg.write(f)
		os.replace(tmp,bin_path)
	def load(self,path):
		root,ext=os.path.splitext(path);bin_path=path if ext.lower()=='.bin'else root+'.bin'
		if not os.path.exists(bin_path):return False
		try:
			with open(bin_path,'rb')as f:msg=_policy_schema.PolicyCheckpoint.read(f)
		except Exception:print('[ML] ckpt .bin corrupto. Se ignora.');return False
		want_obs=int(self.params['W1'].shape[0]);want_hid=int(self.params['W1'].shape[1]);want_na=int(self.na)
		if int(msg.obsDim)!=want_obs or int(msg.hidden)!=want_hid or int(msg.actionCount)!=want_na:print('[ML] ckpt .bin incompatible (obs_dim/hidden/na). Se ignora.');return False
		def _arr2d(t):r=int(t.rows);c=int(t.cols);return jnp.asarray(list(t.data),jnp.float32).reshape((r,c))
		def _arr1d(t):return jnp.asarray(list(t.data),jnp.float32)
		try:W1=_arr2d(msg.w1);b1=_arr1d(msg.b1);Wt=_arr2d(msg.wt);bt=_arr1d(msg.bt);Wp=_arr2d(msg.wp);bp=_arr1d(msg.bp)
		except Exception:print('[ML] ckpt .bin con tensores inválidos. Se ignora.');return False
		new_params={'W1':W1,'b1':b1,'Wt':Wt,'bt':bt,'Wp':Wp,'bp':bp};use_pa=False
		if bool(getattr(msg,'hasPa',False)):
			try:
				Pa=_arr2d(msg.pa);K_loaded,dh_loaded=int(msg.pa.rows),int(msg.pa.cols);AC=self.action_context
				if AC is not None:
					ACf32=jnp.asarray(AC,jnp.float32);K_current=int(ACf32.shape[1]);acid_current=_ac_id_from_AC(ACf32);acid_ckpt=int(getattr(msg,'acId',0))
					if K_loaded==K_current and dh_loaded==want_hid and acid_current is not None and acid_ckpt==acid_current:use_pa=True
					else:print('[ML] Pa(acId/K) incompatible; se ignora Pa del .bin.');use_pa=False
				elif dh_loaded==want_hid:use_pa=True
			except Exception:use_pa=False
		if use_pa:new_params['Pa']=Pa
		self.params=new_params;self.opt_state=self.opt.init(self.params);return True
	def forward_heads(self,obs):
		x=jnp.asarray(obs,jnp.float32);AC=self._ensure_ctx()
		if AC is not None:_,logits,head=_forward_ctx(self.params,x,AC)
		else:_,logits,head=_forward(self.params,x)
		mu01=jnp.clip((jnp.tanh(head[0:2])+1.)/2.,.0,1.);std_xy=jnp.exp(jnp.clip(head[2:4],-3.,1.5));return logits,mu01,std_xy