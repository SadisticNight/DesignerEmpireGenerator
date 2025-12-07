import math,io,capnp,jax.numpy as jnp
from jax import jit
import edificios

cc=capnp.load('celdas.capnp')
sc=capnp.load('stats.capnp')
ED_DATA=edificios.edificios

@jit
def _reduce_stats(res,emp,ind_flag,com_flag,amb,fel):
    te=jnp.sum(emp);tr=jnp.sum(res)
    ti=jnp.sum(emp*ind_flag);tc=jnp.sum(emp*com_flag)
    denom=jnp.maximum(te,1)
    p_ind=ti/denom*1e2;p_com=tc/denom*1e2
    des=te-tr
    eco=jnp.sum(amb);felT=jnp.sum(fel)
    return tr,te,ti,tc,p_ind,p_com,des,eco,felT

def _map_from_bytes(payload):
    try:return cc.Mapa.from_bytes(payload)
    except Exception:return cc.Mapa.read(io.BytesIO(payload))

class StatsProcessor:
    __slots__='map_file','stats_file','providers','resources','seen_hashes'
    
    def __init__(self,map_file='celdas.bin',stats_file='stats.bin'):
        self.map_file=map_file;self.stats_file=stats_file
        self.providers={'energia':('refineria',),'agua':('agua',),'comida':('lecheria',),'basura':('depuradora',)}
        self.resources={'refineria':10000,'agua':40000,'lecheria':4000,'depuradora':40000}
        self.seen_hashes=set()

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
        KNOWN=set(ED_DATA.keys())|{'suelo','decoracion'}
        self.seen_hashes.clear()
        stats=sc.Stats.new_message()
        counts={}
        
        L_hash=[];L_res=[];L_emp=[];L_ind=[];L_com=[];L_amb=[];L_fel=[]
        L_enr=[];L_agu=[];L_comi=[];L_bas=[]
        
        enrT=aguT=comT=basT=0
        
        sset=self.seen_hashes
        seen_iids=set()
        mapa=self._load_map()
        
        if not hasattr(mapa,'celdas'):mapa=cc.Mapa.new_message()
        
        for c in mapa.celdas:
            h=self._u64(c.hash)
            if not h or h in sset:continue
            sset.add(h)
            
            bn=self._canon(c.edificio)
            
            if bn in ED_DATA:
                defi=ED_DATA[bn]
                res_val=int(defi.residentes)
                emp_val=int(defi.empleos)
                fel_val=int(defi.felicidad)
                amb_val=int(defi.ambiente)
                
                e_val=int(defi.energia)
                if e_val>0:enr_cap=e_val;e_use=0
                else:enr_cap=0;e_use=e_val
                
                a_val=int(defi.agua)
                if a_val>0:agu_cap=a_val;a_use=0
                else:agu_cap=0;a_use=a_val
                
                c_val=int(defi.comida)
                if c_val>0:com_cap=c_val;c_use=0
                else:com_cap=0;c_use=c_val
                
                b_val=int(defi.basura)
                if b_val>0:bas_cap=b_val;b_use=0
                else:bas_cap=0;b_use=b_val
                
                tipo_str=str(getattr(defi.tipo,'value',defi.tipo)).lower()
                is_ind=1 if 'industria' in tipo_str else 0
                is_com=1 if 'comercio' in tipo_str else 0
                
            else:
                a=c.atributos
                t=self._canon(c.tipo)
                res_val=a.residentes
                emp_val=a.empleos
                fel_val=a.felicidad
                amb_val=a.ambiente
                e_use=a.energia
                a_use=a.agua
                c_use=a.comida
                b_use=a.basura
                enr_cap=agu_cap=com_cap=bas_cap=0
                is_ind=1 if t=='industria' else 0
                is_com=1 if t=='comercio' else 0

            L_hash.append(h&4294967295)
            L_res.append(res_val)
            L_emp.append(emp_val)
            L_ind.append(is_ind)
            L_com.append(is_com)
            L_amb.append(amb_val)
            L_fel.append(fel_val)
            L_enr.append(e_use) 
            L_agu.append(a_use)
            L_comi.append(c_use)
            L_bas.append(b_use)
            
            iid=h>>8&0xffffffffffffffff
            if iid not in seen_iids:
                if bn in ED_DATA:
                    d=ED_DATA[bn]
                    if d.energia>0:enrT+=d.energia
                    if d.agua>0:aguT+=d.agua
                    if d.comida>0:comT+=d.comida
                    if d.basura>0:basT+=d.basura
                
                counts[bn]=counts.get(bn,0)+1
                seen_iids.add(iid)

        if L_res:
            H=jnp.array(L_hash,dtype=jnp.uint32)
            RES=jnp.array(L_res,dtype=jnp.int32)
            EMP=jnp.array(L_emp,dtype=jnp.int32)
            IND=jnp.array(L_ind,dtype=jnp.int32)
            COM=jnp.array(L_com,dtype=jnp.int32)
            AMB=jnp.array(L_amb,dtype=jnp.int32)
            FEL=jnp.array(L_fel,dtype=jnp.int32)
            E=jnp.array(L_enr,dtype=jnp.int32)
            A=jnp.array(L_agu,dtype=jnp.int32)
            C=jnp.array(L_comi,dtype=jnp.int32)
            B=jnp.array(L_bas,dtype=jnp.int32)
            
            ACTIVE=jnp.ones_like(RES,dtype=bool)

            RES=RES*ACTIVE;EMP=EMP*ACTIVE;FEL=FEL*ACTIVE
            
            tr,te,ti,tc,pind,pcom,des,eco,felT=_reduce_stats(RES,EMP,IND,COM,AMB,FEL)
            
            tr=int(tr);te=int(te);ti=int(ti);tc=int(tc)
            des=int(des);eco=int(eco);felT=int(felT)
            pind=float(pind);pcom=float(pcom)
            
            enrU=int(jnp.sum(E)) 
            aguU=int(jnp.sum(A))
            comU=int(jnp.sum(C))
            basU=int(jnp.sum(B))
            
        else:
            tr=te=ti=tc=des=eco=felT=0
            pind=pcom=.0
            enrU=aguU=comU=basU=0

        stats.totalResidentes=tr
        stats.totalEmpleos=te
        stats.totalEmpleosIndustria=ti
        stats.totalEmpleosComercio=tc
        stats.porcentajeIndustria=pind
        stats.porcentajeComercio=pcom
        stats.desequilibrioLaboral=des
        
        try:stats.ec=math.isclose(ti/(te or 1),1/3)and math.isclose(tc/(te or 1),2/3)and abs(des)<=100
        except Exception:pass
        
        stats.energiaUsada=enrU;stats.energiaTotal=enrT
        stats.aguaUsada=aguU;stats.aguaTotal=aguT
        stats.comidaUsada=comU;stats.comidaTotal=comT
        stats.basuraUsada=basU;stats.basuraTotal=basT
        
        stats.ecologiaTotal=eco
        stats.felicidadTotal=felT
        
        lst=stats.init('cantidadEdificios',len(counts))
        for(i,(name,qty))in enumerate(counts.items()):
            e=lst[i];e.nombre=name;e.cantidad=qty
            
        return stats

if __name__=='__main__':
    st=StatsProcessor().process()
    out=[f"Residentes: {st.totalResidentes}",f"Empleos: {st.totalEmpleos}",
         f"  Industria: {st.totalEmpleosIndustria} ({st.porcentajeIndustria:.1f}%)",
         f"  Comercio: {st.totalEmpleosComercio} ({st.porcentajeComercio:.1f}%)",
         f"Desbalance: {st.desequilibrioLaboral}",
         f"Energía R/T: {st.energiaUsada}/{st.energiaTotal}",
         f"Agua R/T: {st.aguaUsada}/{st.aguaTotal}",
         f"Comida R/T: {st.comidaUsada}/{st.comidaTotal}",
         f"Basura R/T: {st.basuraUsada}/{st.basuraTotal}",
         f"Ecología: {st.ecologiaTotal}",f"Felicidad: {st.felicidadTotal}"]
    print('\n'.join(out))