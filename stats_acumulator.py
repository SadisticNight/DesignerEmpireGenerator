# stats_acumulator.py
import capnp,math
from typing import Dict,Tuple,Any,Optional,Set,Iterable
import edificios
sc=capnp.load('stats.capnp')
ED=edificios.edificios
def _tipo_str(e):return str(getattr(e,'tipo',getattr(e,'tipo',''))if not hasattr(e,'tipo')else getattr(e.tipo,'value',e.tipo)).lower()
class StatsAccumulator:
	__slots__='cnt','res_base','emp_base','emp_ind_base','emp_com_base','amb_base','fel_base','pos','cons'
	def __init__(self):self.reset()
	def reset(self):self.cnt={};self.res_base=0;self.emp_base=0;self.emp_ind_base=0;self.emp_com_base=0;self.amb_base=0;self.fel_base=0;self.pos={'energia':0,'agua':0,'comida':0,'basura':0};self.cons={'energia':0,'agua':0,'comida':0,'basura':0}
	def _apply_bn(self,bn,sign):
		e=ED.get(bn);s=int(sign)
		def _noop():0
		def _do():
			nprev=self.cnt.get(bn,0);nnew=nprev+s;nnew>0 and(self.cnt.__setitem__(bn,nnew)or True)or self.cnt.pop(bn,None);emp=int(getattr(e,'empleos',0));self.res_base+=s*int(getattr(e,'residentes',0));self.emp_base+=s*emp;t=_tipo_str(e);self.emp_ind_base+=s*emp*int(t=='industria');self.emp_com_base+=s*emp*int(t=='comercio');self.amb_base+=s*int(getattr(e,'ambiente',0));self.fel_base+=s*int(getattr(e,'felicidad',0))
			for r in('energia','agua','comida','basura'):v=int(getattr(e,r,0));self.pos[r]+=s*max(0,v);self.cons[r]+=s*max(0,-v)
		return{True:_do,False:_noop}[bool(e)and s!=0]()
	def add_building(self,bn,k=1):self._apply_bn(bn,+int(k))
	def remove_building(self,bn,k=1):self._apply_bn(bn,-int(k))
	def rebuild_from_board(self,board):
		self.reset();seen=set()
		for v in board.values():
			def _tuple_case():bn,iid=v;key=bn,int(iid);added=key not in seen;seen.add(key);added and self._apply_bn(bn,+1)
			def _simple_case():self._apply_bn(str(v),+1)
			{True:_tuple_case,False:_simple_case}[isinstance(v,tuple)]()
	@staticmethod
	def _frac_keep(pos,cons):c=float(cons);has_cons=float(c>.0);denom=max(c,1.);ratio=float(pos)/denom;clamped=max(.0,min(1.,ratio));return(1.-has_cons)*1.+has_cons*clamped
	def to_stats(self):
		stats=sc.Stats.new_message();fE=self._frac_keep(self.pos['energia'],self.cons['energia']);fA=self._frac_keep(self.pos['agua'],self.cons['agua']);fC=self._frac_keep(self.pos['comida'],self.cons['comida']);fB=self._frac_keep(self.pos['basura'],self.cons['basura']);fALL=fE*fA*fC*fB;tr=int(self.res_base*fALL);te=int(self.emp_base*fALL);ti=int(self.emp_ind_base*fALL);tc=int(self.emp_com_base*fALL);eco=int(self.amb_base*fALL);fel=int(self.fel_base*fALL);denom=max(te,1);p_ind=float(ti)/float(denom)*1e2;p_com=float(tc)/float(denom)*1e2;des=int(te-tr);enrU=int(self.pos['energia']-fE*self.cons['energia']);aguU=int(self.pos['agua']-fA*self.cons['agua']);comU=int(self.pos['comida']-fC*self.cons['comida']);basU=int(self.pos['basura']-fB*self.cons['basura']);enrT=int(self.pos['energia']);aguT=int(self.pos['agua']);comT=int(self.pos['comida']);basT=int(self.pos['basura']);stats.totalResidentes=tr;stats.totalEmpleos=te;stats.totalEmpleosIndustria=ti;stats.totalEmpleosComercio=tc;stats.porcentajeIndustria=float(p_ind);stats.porcentajeComercio=float(p_com);stats.desequilibrioLaboral=des;stats.ec=math.isclose(ti/(te or 1),1/3)and math.isclose(tc/(te or 1),2/3)and abs(des)<=100;stats.energiaUsada=enrU;stats.energiaTotal=enrT;stats.aguaUsada=aguU;stats.aguaTotal=aguT;stats.comidaUsada=comU;stats.comidaTotal=comT;stats.basuraUsada=basU;stats.basuraTotal=basT;stats.ecologiaTotal=eco;stats.felicidadTotal=fel;lst=stats.init('cantidadEdificios',len(self.cnt))
		for(i,(name,qty))in enumerate(self.cnt.items()):e=lst[i];e.nombre=name;e.cantidad=int(qty)
		return stats
	def clear_and_add(self,items):
		self.reset()
		for(bn,k)in items:self._apply_bn(bn,+int(k))
	def snapshot(self):return dict(self.cnt)
if __name__=='__main__':acc=StatsAccumulator();acc.add_building('refineria',2);acc.add_building('agua',1);acc.add_building('lecheria',1);acc.add_building('depuradora',1);acc.add_building('residencia',10);st=acc.to_stats();out=[f"Residentes: {st.totalResidentes}",f"Empleos: {st.totalEmpleos}",f"Industria/Comercio: {st.totalEmpleosIndustria}/{st.totalEmpleosComercio}  ({st.porcentajeIndustria:.1f}% / {st.porcentajeComercio:.1f}%)",f"Desbalance: {st.desequilibrioLaboral}",f"Energía R/T: {st.energiaUsada}/{st.energiaTotal}",f"Agua R/T:    {st.aguaUsada}/{st.aguaTotal}",f"Comida R/T:  {st.comidaUsada}/{st.comidaTotal}",f"Basura R/T:  {st.basuraUsada}/{st.basuraTotal}",f"Ecología: {st.ecologiaTotal}",f"Felicidad: {st.felicidadTotal}",f"EC flag: {st.ec}"];print('\n'.join(out))