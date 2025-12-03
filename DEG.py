# DEG.py
import os,sys,importlib
_BASE=os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)
def _ensure(bin_name,mod,fn):
	if not os.path.exists(bin_name):getattr(importlib.import_module(mod),fn)()
_ensure('celdas.bin','generar_celdas','generar_datos')
_ensure('estadisticas.bin','generar_stats','generar_stats')
_ensure('edificios.bin','generar_edificios','generar_edificios')
from juego import Game
def main():Game().bucle_principal()
if __name__=='__main__':main()