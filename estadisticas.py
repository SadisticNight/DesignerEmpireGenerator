# estadisticas.py
import math,io,capnp,jax.numpy as jnp
from jax import jit
cc=capnp.load('celdas.capnp')
sc=capnp.load('stats.capnp')
@jit
def _reduce_stats(res,emp,ind_flag,com_flag,amb,fel):te=jnp.sum(emp);tr=jnp.sum(res);ti=jnp.sum(emp*ind_flag);tc=jnp.sum(emp*com_flag);denom=jnp.maximum(te,1);p_ind=ti/denom*1e2;p_com=tc/denom*1e2;des=te-tr;eco=jnp.sum(amb);felT=jnp.sum(fel);return tr,te,ti,tc,p_ind,p_com,des,eco,felT
def _map_from_bytes(payload):
	try:return cc.Mapa.from_bytes(payload)
	except Exception:return cc.Mapa.read(io.BytesIO(payload))
class StatsProcessor:
	__slots__='map_file','stats_file','providers','resources','seen_hashes'
	def __init__(self,map_file='celdas.bin',stats_file='stats.bin'):self.map_file=map_file;self.stats_file=stats_file;self.providers={'energia':('refineria',),'agua':('agua',),'comida':('lecheria',),'basura':('depuradora',)};self.resources={'refineria':10000,'agua':40000,'lecheria':4000,'depuradora':40000};self.seen_hashes=set()
	def _load_map(self):
		try:
			import bd_celdas;get_msg=getattr(bd_celdas,'get_snapshot_msg',None)
			if callable(get_msg):
				msg=get_msg()
				if msg is not None:return msg
			get_bytes=getattr(bd_celdas,'get_snapshot_bytes',None)
			if callable(get_bytes):
				payload=get_bytes()
				if isinstance(payload,(bytes,bytearray,memoryview)):return _map_from_bytes(bytes(payload))
		except Exception:pass
		try:
			with open(self.map_file,'rb')as f:return cc.Mapa.read(f)
		except FileNotFoundError:return cc.Mapa.new_message()
		except Exception:return cc.Mapa.new_message()
	@staticmethod
	def _u64(h):
		try:
			if isinstance(h,int):return h&0xffffffffffffffff
			if isinstance(h,(bytes,bytearray)):return int.from_bytes(bytes(h[:8]),'little',signed=False)
			if isinstance(h,str):
				hs=h.strip()
				if not hs:return 0
				try:return int(hs,16)&0xffffffffffffffff
				except ValueError:return int(hs,10)&0xffffffffffffffff
			return int(h)&0xffffffffffffffff
		except Exception:return 0
	@staticmethod
	def _canon(s):
		try:return s.strip().lower()
		except Exception:return str(s).lower()
	def process(self):
		from edificios import edificios as ED;KNOWN=set(ED.keys())|{'suelo','decoracion'};self.seen_hashes.clear();stats=sc.Stats.new_message();counts={};L_hash=[];L_res=[];L_emp=[];L_ind=[];L_com=[];L_amb=[];L_fel=[];L_enr=[];L_agu=[];L_comi=[];L_bas=[];enrT=aguT=comT=basT=0;prov=self.providers;cap=self.resources;sset=self.seen_hashes;seen_iids=set();mapa=self._load_map()
		if not hasattr(mapa,'celdas'):mapa=cc.Mapa.new_message()
		for c in mapa.celdas:
			h=self._u64(c.hash)
			if not h or h in sset:continue
			sset.add(h);a=c.atributos;bn=self._canon(c.edificio);t=self._canon(c.tipo);L_hash.append(h&4294967295);L_res.append(a.residentes);L_emp.append(a.empleos);L_ind.append(1 if t=='industria'else 0);L_com.append(1 if t=='comercio'else 0);has_edif=bn in KNOWN or t in KNOWN or bool(a.residentes or a.empleos);L_amb.append(a.ambiente if has_edif else 0);L_fel.append(a.felicidad);L_enr.append(a.energia);L_agu.append(a.agua);L_comi.append(a.comida);L_bas.append(a.basura);iid=h>>8&0xffffffffffffffff
			if iid not in seen_iids:
				if bn in prov['energia']:enrT+=cap.get(bn,0)
				if bn in prov['agua']:aguT+=cap.get(bn,0)
				if bn in prov['comida']:comT+=cap.get(bn,0)
				if bn in prov['basura']:basT+=cap.get(bn,0)
				counts[bn]=counts.get(bn,0)+1;seen_iids.add(iid)
		if L_res:
			H=jnp.array(L_hash,dtype=jnp.uint32);RES=jnp.array(L_res,dtype=jnp.int32);EMP=jnp.array(L_emp,dtype=jnp.int32);IND=jnp.array(L_ind,dtype=jnp.int32);COM=jnp.array(L_com,dtype=jnp.int32);AMB=jnp.array(L_amb,dtype=jnp.int32);FEL=jnp.array(L_fel,dtype=jnp.int32);E=jnp.array(L_enr,dtype=jnp.int32);A=jnp.array(L_agu,dtype=jnp.int32);C=jnp.array(L_comi,dtype=jnp.int32);B=jnp.array(L_bas,dtype=jnp.int32)
			def _mask_deficit(x,H_):total=jnp.sum(x).astype(jnp.float32);cons=jnp.sum(jnp.abs(jnp.minimum(x,0))).astype(jnp.float32);need=jnp.maximum(-total,jnp.float32(.0));cons=jnp.maximum(cons,jnp.float32(1.));keep=jnp.float32(1.)-need/cons;keep=jnp.clip(keep,jnp.float32(.0),jnp.float32(1.));hf=H_.astype(jnp.float32);r=jnp.sin(hf*jnp.float32(12.9898)+jnp.float32(78.233))*jnp.float32(43758.5453);r=r-jnp.floor(r);return(x>=0)|(r<keep)
			mE=_mask_deficit(E,H);mA=_mask_deficit(A,H);mC=_mask_deficit(C,H);mB=_mask_deficit(B,H);ACTIVE=mE&mA&mC&mB;RES=RES*ACTIVE;EMP=EMP*ACTIVE;FEL=FEL*ACTIVE;E=jnp.where(E<0,E*ACTIVE,E);A=jnp.where(A<0,A*ACTIVE,A);C=jnp.where(C<0,C*ACTIVE,C);B=jnp.where(B<0,B*ACTIVE,B);tr,te,ti,tc,pind,pcom,des,eco,felT=_reduce_stats(RES,EMP,IND,COM,AMB,FEL);tr=int(tr);te=int(te);ti=int(ti);tc=int(tc);des=int(des);eco=int(eco);felT=int(felT);pind=float(pind);pcom=float(pcom);consE=int(jnp.sum(jnp.abs(jnp.minimum(E,0))));consA=int(jnp.sum(jnp.abs(jnp.minimum(A,0))));consC=int(jnp.sum(jnp.abs(jnp.minimum(C,0))));consB=int(jnp.sum(jnp.abs(jnp.minimum(B,0))));enrU=enrT-consE;aguU=aguT-consA;comU=comT-consC;basU=basT-consB
		else:tr=te=ti=tc=des=eco=felT=0;pind=pcom=.0;enrU=aguU=comU=basU=0
		stats.totalResidentes=tr;stats.totalEmpleos=te;stats.totalEmpleosIndustria=ti;stats.totalEmpleosComercio=tc;stats.porcentajeIndustria=pind;stats.porcentajeComercio=pcom;stats.desequilibrioLaboral=des
		try:stats.ec=math.isclose(ti/(te or 1),1/3)and math.isclose(tc/(te or 1),2/3)and abs(des)<=100
		except Exception:pass
		stats.energiaUsada=enrU;stats.energiaTotal=enrT;stats.aguaUsada=aguU;stats.aguaTotal=aguT;stats.comidaUsada=comU;stats.comidaTotal=comT;stats.basuraUsada=basU;stats.basuraTotal=basT;stats.ecologiaTotal=eco;stats.felicidadTotal=felT;lst=stats.init('cantidadEdificios',len(counts))
		for(i,(name,qty))in enumerate(counts.items()):e=lst[i];e.nombre=name;e.cantidad=qty
		return stats
if __name__=='__main__':st=StatsProcessor().process();out=[f"Residentes: {st.totalResidentes}",f"Empleos: {st.totalEmpleos}",f"  Industria: {st.totalEmpleosIndustria} ({st.porcentajeIndustria:.1f}%)",f"  Comercio: {st.totalEmpleosComercio} ({st.porcentajeComercio:.1f}%)",f"Desbalance: {st.desequilibrioLaboral}",f"Energía R/T: {st.energiaUsada}/{st.energiaTotal}",f"Agua R/T: {st.aguaUsada}/{st.aguaTotal}",f"Comida R/T: {st.comidaUsada}/{st.comidaTotal}",f"Basura R/T: {st.basuraUsada}/{st.basuraTotal}",f"Ecología: {st.ecologiaTotal}",f"Felicidad: {st.felicidadTotal}"];print('\n'.join(out))
