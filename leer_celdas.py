# leer_celdas.py
import sys, argparse, capnp
cc = capnp.load('celdas.capnp')

def _load(fn):
    try:
        with open(fn,'rb') as f: return cc.Mapa.read(f)
    except Exception as e:
        print(f"Error: {e}"); sys.exit(1)

def _has_core(c, with_env=False, with_services=False):
    if c.edificio: return True
    a=c.atributos
    if (a.energia|a.agua|a.basura|a.comida|a.empleos|a.residentes)!=0: return True
    if with_env and (a.felicidad|a.ambiente)!=0: return True
    if with_services:
        s=c.servicios
        if (s.seguridad|s.incendio|s.salud|s.educacion)!=0: return True
    return False

def _fmt(c):
    a=c.atributos; s=c.servicios
    return ("{"
            f"'x':{c.x},'y':{c.y},'edificio':'{c.edificio}','tipo':'{c.tipo}',"
            f"'atributos':{{'energia':{a.energia},'agua':{a.agua},'basura':{a.basura},'comida':{a.comida},"
            f"'empleos':{a.empleos},'residentes':{a.residentes},'felicidad':{a.felicidad},'ambiente':{a.ambiente}}},"
            f"'servicios':{{'seguridad':{s.seguridad},'incendio':{s.incendio},'salud':{s.salud},'educacion':{s.educacion}}}"
            "}")

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("bin", nargs="?", default="celdas.bin")
    ap.add_argument("--with-env", action="store_true")
    ap.add_argument("--with-services", action="store_true")
    o=ap.parse_args()

    m=_load(o.bin); cnt=0
    for c in m.celdas:
        if _has_core(c, o.with_env, o.with_services):
            print(_fmt(c)); cnt+=1
    print(f"Total celdas con datos: {cnt}")

if __name__=="__main__": main()
