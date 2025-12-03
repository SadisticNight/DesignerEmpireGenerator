# juego.py
import pygame,sys,signal,threading,time,traceback
from collections import deque
from config import BOARD_SIZE,WINDOW_SIZE,CELL_SIZE,BACKGROUND_COLOR,PLAYER_COLOR,PANEL_TOP_H,PANEL_BOTTOM_H,STATS_PANEL_W,STATS_PANEL_PADDING
from menu import Menu
from modo_manual import ModoManual
from modo_automatico import ModoAutomatico
from estadisticas import StatsProcessor
import edificios,bd_celdas
_shutdown_event=threading.Event()
def _signal_handler(sig,frame):
	_shutdown_event.set()
	try:
		if pygame.get_init():pygame.event.post(pygame.event.Event(pygame.QUIT))
	except Exception as e:print('[WARN] signal handler post QUIT failed:',e,file=sys.stderr);traceback.print_exc()
signal.signal(signal.SIGINT,_signal_handler)
try:signal.signal(signal.SIGTERM,_signal_handler)
except Exception:pass
EVT_CITY_CHANGED=pygame.USEREVENT+1
EVT_HUD=pygame.USEREVENT+2
P1H=PANEL_TOP_H
P3H=PANEL_BOTTOM_H
BXS=20
BSZ=128,36
COL_P1=BACKGROUND_COLOR
COL_P3=BACKGROUND_COLOR
COL_SEP=60,60,60
COL_TEXT=230,230,230
COL_BTN=55,55,55
COL_BH=75,75,75
COL_BB=90,90,90
COL_ACTIVE=76,194,255
STATS_BG=BACKGROUND_COLOR
COL_DIM=70,70,70
LOCK_PROVIDERS={'energia':'refineria','agua':'agua','comida':'lecheria','basura':'depuradora'}
t_stats=[]
t_stats_surfaces=[]
board={}
_BOARD_LOCK=None
player_pos=[BOARD_SIZE//2,BOARD_SIZE//2]
_last_player_pos=tuple(player_pos)
selected_building=None
auto_mode=False
lock_resources_ui=set()
HIGHLIGHTS=deque(maxlen=64)
HIGHLIGHT_MS=950
NS_PER_MS=1000000
_DIRTY_CELLS=set()
_FORCE_FULL_REPAINT=True
def _button_rect():return pygame.Rect(BXS,(P1H-BSZ[1])//2,*BSZ)
def _u64(h):
	try:
		if isinstance(h,int):return h&0xffffffffffffffff
		if isinstance(h,bytes):return int.from_bytes(h[:8],'little',signed=False)
		if isinstance(h,str):
			hs=h.strip()
			if not hs:return 0
			try:return int(hs,16)&0xffffffffffffffff
			except ValueError:return int(hs,10)&0xffffffffffffffff
		return int(h)&0xffffffffffffffff
	except Exception:return 0
def _is_anchor(h):h=_u64(h);return bool(h)and h&255==0
def _calc_lock_resources(s):return{k for(k,v)in(('energia',s.energiaUsada),('agua',s.aguaUsada),('comida',s.comidaUsada),('basura',s.basuraUsada))if v<0}
def _apply_lock_and_stats():
	bd_celdas.update_celdas_bin(board,lock_resources=set());s1=StatsProcessor().process();lr=_calc_lock_resources(s1)
	if lr:bd_celdas.update_celdas_bin(board,lock_resources=lr);s1=StatsProcessor().process()
	return lr,s1
def _fmt_stats(s):return[f"Residentes: {s.totalResidentes}",f"Empleos: {s.totalEmpleos}",f"  Industria: {s.totalEmpleosIndustria} ({s.porcentajeIndustria:.1f}%)",f"  Comercio: {s.totalEmpleosComercio} ({s.porcentajeComercio:.1f}%)",f"Desbalance: {s.desequilibrioLaboral}",f"Energía R/T: {s.energiaUsada}/{s.energiaTotal}",f"Agua R/T: {s.aguaUsada}/{s.aguaTotal}",f"Comida R/T: {s.comidaUsada}/{s.comidaTotal}",f"Basura R/T: {s.basuraUsada}/{s.basuraTotal}",f"Ecología: {s.ecologiaTotal}",f"Felicidad: {s.felicidadTotal}"]
def _update_stats_surfaces(font):global t_stats_surfaces;t_stats_surfaces=[font.render(line,True,COL_TEXT)for line in t_stats]
ALLOW_STATIC={'suelo','decoracion'}
COLOR_DEFAULT=100,100,100
def _cell_rect(x,y):cs,y0=CELL_SIZE,PANEL_TOP_H;return pygame.Rect(x*cs,y*cs+y0,cs,cs)
def _mark_dirty_rect(x,y,w,h):
	for i in range(w):
		for j in range(h):_DIRTY_CELLS.add((x+i,y+j))
def _snapshot_items_for_cells(cells):
	lk=globals().get('_BOARD_LOCK')
	if lk:
		with lk:return[((x,y),board.get((x,y)))for(x,y)in cells]
	else:return[((x,y),board.get((x,y)))for(x,y)in cells]
def _snapshot_all_items():
	lk=globals().get('_BOARD_LOCK')
	if lk:
		with lk:return tuple(board.items())
	return tuple(board.items())
def dibujar_tablero(p,show_player,*,dirty_only):
	cs,y0=CELL_SIZE,P1H;e=edificios.edificios;rect=pygame.draw.rect;allow={LOCK_PROVIDERS[r]for r in lock_resources_ui if r in LOCK_PROVIDERS}if lock_resources_ui else None
	if not dirty_only:
		p.fill(BACKGROUND_COLOR);items=_snapshot_all_items()
		for((x,y),b)in items:name=b[0]if isinstance(b,tuple)else b;low=(name or'').lower();dim=bool(allow)and low not in allow and low not in ALLOW_STATIC;col=COL_DIM if dim else e[name].color if name in e else COLOR_DEFAULT;rect(p,col,(x*cs,y*cs+y0,cs,cs))
	else:
		wanted=set(_DIRTY_CELLS)
		if not wanted:return
		samples=_snapshot_items_for_cells(wanted)
		for((x,y),b)in samples:
			rect(p,BACKGROUND_COLOR,(x*cs,y*cs+y0,cs,cs))
			if b is not None:name=b[0]if isinstance(b,tuple)else b;low=(name or'').lower();dim=bool(allow)and low not in allow and low not in ALLOW_STATIC;col=COL_DIM if dim else e[name].color if name in e else COLOR_DEFAULT;rect(p,col,(x*cs,y*cs+y0,cs,cs))
		_DIRTY_CELLS.clear()
	if show_player:rect(p,PLAYER_COLOR,(player_pos[0]*cs,player_pos[1]*cs+y0,cs,cs))
def _draw_highlights(p,now_ns):
	if not HIGHLIGHTS:return
	cs,y0=CELL_SIZE,P1H;e=edificios.edificios;keep=[];draw_rect=pygame.draw.rect
	for(bn,x,y,w,h,t_end_ns,kind,hv)in list(HIGHLIGHTS):
		if now_ns>t_end_ns:continue
		if hv is not None and not _is_anchor(hv):continue
		col=(220,80,80)if kind=='demolished'else e[bn].color if bn in e else COL_ACTIVE;draw_rect(p,col,pygame.Rect(x*cs,y*cs+y0,w*cs,h*cs),width=2);keep.append((bn,x,y,w,h,t_end_ns,kind,hv))
	HIGHLIGHTS.clear();HIGHLIGHTS.extend(keep)
def draw_button(p,active,enabled=True):r=_button_rect();hov=enabled and r.collidepoint(pygame.mouse.get_pos());base=COL_BH if hov else COL_BTN;pygame.draw.rect(p,base,r,border_radius=4);pygame.draw.rect(p,COL_BB,r,1,border_radius=4);color_txt=(160,160,160)if not enabled else COL_ACTIVE if active else COL_TEXT;t=pygame.font.SysFont(None,18).render('Auto Mode',True,color_txt);p.blit(t,(r.x+(BSZ[0]-t.get_width())//2,r.y+(BSZ[1]-t.get_height())//2))
def dibujar_ui(p,boton_habilitado=True):pygame.draw.rect(p,COL_P1,(0,0,WINDOW_SIZE,P1H));pygame.draw.line(p,COL_SEP,(0,P1H-1),(WINDOW_SIZE,P1H-1));draw_button(p,auto_mode,enabled=boton_habilitado);lbl=pygame.font.SysFont(None,18).render(f"Edif seleccionado: {selected_building or"Ninguno"}",True,COL_TEXT);p.blit(lbl,(_button_rect().right+20,(P1H-lbl.get_height())//2));title=pygame.font.SysFont(None,18).render('Estadísticas',True,COL_TEXT);p.blit(title,(WINDOW_SIZE+STATS_PANEL_PADDING,(P1H-title.get_height())//2))
def dibujar_panel_fino(p):pygame.draw.rect(p,COL_P3,(0,WINDOW_SIZE+P1H,WINDOW_SIZE,P3H))
def _set_auto_mode(enable,menu):
	global auto_mode,_FORCE_FULL_REPAINT;auto_mode=enable;pygame.mouse.set_visible(True);_FORCE_FULL_REPAINT=True;fn=getattr(menu,'set_enabled',None)
	try:fn and fn(not enable)
	except Exception as e:print('[WARN] menu.set_enabled failed:',e,file=sys.stderr);traceback.print_exc()
	fnc=getattr(menu,'cerrar',None)
	if enable and fnc:
		try:fnc()
		except Exception as e:print('[WARN] menu.cerrar failed:',e,file=sys.stderr);traceback.print_exc()
def main():
	global t_stats,t_stats_surfaces,selected_building,lock_resources_ui,_BOARD_LOCK,_FORCE_FULL_REPAINT,_last_player_pos;pygame.init();flags=pygame.HWSURFACE|pygame.DOUBLEBUF;w,h=WINDOW_SIZE+STATS_PANEL_W,WINDOW_SIZE+P1H+P3H
	try:screen=pygame.display.set_mode((w,h),flags,vsync=1)
	except TypeError:screen=pygame.display.set_mode((w,h),flags)
	pygame.display.set_caption('DEG - CityBuilder');clock=pygame.time.Clock();font=pygame.font.SysFont(None,16);pygame.mouse.set_visible(True);pygame.event.set_allowed([pygame.QUIT,pygame.KEYDOWN,pygame.MOUSEBUTTONDOWN,EVT_HUD,EVT_CITY_CHANGED]);ev_get=pygame.event.get;monotonic_ns=time.monotonic_ns;draw_tab=dibujar_tablero;draw_ui=dibujar_ui;draw_panel=dibujar_panel_fino;draw_hl=_draw_highlights;lock=threading.Lock();_BOARD_LOCK=lock;menu=Menu();manual=ModoManual(board,player_pos,lock=lock);auto=ModoAutomatico(board,flush_every=1);auto.lock=lock
	def _auto_hud(kind,bn,x,y,w_,h_,*rest,**kw):
		hv=kw.get('h',None);hv=rest[0]if hv is None and rest else hv
		try:pygame.event.post(pygame.event.Event(EVT_HUD,{'kind':kind,'bn':bn,'x':x,'y':y,'w':w_,'h':h_,'hv':hv}))
		except Exception as e:print('[WARN] post EVT_HUD failed:',e,file=sys.stderr);traceback.print_exc()
	auto.on_change=_auto_hud
	def _cleanup_and_exit():
		try:auto.save_now()
		except Exception as e:
			print('[WARN] auto.save_now() failed:',e,file=sys.stderr);traceback.print_exc()
			try:
				auto.running=False
				if getattr(auto,'thread',None):auto.thread.join(timeout=.5)
			except Exception as e2:print('[WARN] auto thread join failed:',e2,file=sys.stderr);traceback.print_exc()
		try:pygame.quit()
		finally:sys.exit(0)
	try:
		with lock:s=auto.last_stats
		if not s:
			bd_celdas.update_celdas_bin(board,lock_resources=set());s=StatsProcessor().process();lock_resources_ui=_calc_lock_resources(s)
			if lock_resources_ui:bd_celdas.update_celdas_bin(board,lock_resources=lock_resources_ui);s=StatsProcessor().process()
		else:lock_resources_ui=getattr(auto,'last_lock_resources',set())
		t_stats=_fmt_stats(s);_update_stats_surfaces(font);_FORCE_FULL_REPAINT=True
	except Exception as e:print('[FATAL] warm-up stats failed:',e,file=sys.stderr);traceback.print_exc();_cleanup_and_exit()
	def _on_quit(e):_cleanup_and_exit()
	def _on_hud(e):
		ed=getattr(e,'dict',e.__dict__);bn=ed.get('bn','?');x=int(ed.get('x',0));y=int(ed.get('y',0));w_=int(ed.get('w')or 1);h_=int(ed.get('h')or 1);kind=ed.get('kind','placed');hv=ed.get('hv',None)
		if hv is not None and not _is_anchor(hv):return
		now_ns=monotonic_ns();HIGHLIGHTS.append((bn,x,y,w_,h_,now_ns+HIGHLIGHT_MS*NS_PER_MS,kind,hv));_mark_dirty_rect(x,y,w_,h_)
	def _on_city_changed(e):
		global lock_resources_ui,t_stats,_FORCE_FULL_REPAINT
		with lock:lr,s=_apply_lock_and_stats()
		lock_resources_ui=lr;t_stats=_fmt_stats(s);_update_stats_surfaces(font);_FORCE_FULL_REPAINT=True;manual.updated=False
	def _toggle_auto():_set_auto_mode(not auto_mode,menu)
	def _key_default(e):
		global selected_building
		if auto_mode:return
		if menu.esta_abierto():
			op=menu.manejar_evento(e)
			if op:selected_building=op
		else:
			sel=manual.handle_event(e,menu,selected_building)
			if sel!=selected_building:selected_building=sel
	key_handlers={pygame.K_a:_toggle_auto}
	def _on_keydown(e):(key_handlers.get(e.key)or _key_default)(e)
	def _on_mouse(e):
		if _button_rect().collidepoint(e.pos):_set_auto_mode(not auto_mode,menu)
	def _noop(e):0
	handlers={pygame.QUIT:_on_quit,EVT_HUD:_on_hud,EVT_CITY_CHANGED:_on_city_changed,pygame.KEYDOWN:_on_keydown,pygame.MOUSEBUTTONDOWN:_on_mouse}
	try:
		while True:
			if _shutdown_event.is_set():_cleanup_and_exit()
			clock.tick(60)
			if _last_player_pos!=tuple(player_pos):ox,oy=_last_player_pos;_mark_dirty_rect(ox,oy,1,1);nx,ny=player_pos;_mark_dirty_rect(nx,ny,1,1);_last_player_pos=tuple(player_pos)
			for e in ev_get():
				try:(handlers.get(e.type)or _noop)(e)
				except Exception as ex:print('[FATAL] event dispatch failed:',ex,file=sys.stderr);traceback.print_exc();_cleanup_and_exit()
			if not auto_mode and not menu.esta_abierto():manual.update_movement()
			auto.actualizar(1 if auto_mode else 0,0)
			if manual.updated:
				try:
					with lock:lr,s=_apply_lock_and_stats()
					lock_resources_ui=lr;t_stats=_fmt_stats(s);_update_stats_surfaces(font);manual.updated=False;_FORCE_FULL_REPAINT=True
				except Exception as e3:print('[FATAL] manual.updated apply failed:',e3,file=sys.stderr);traceback.print_exc();_cleanup_and_exit()
			if auto.updated:
				try:s=auto.last_stats;lock_resources_ui=getattr(auto,'last_lock_resources',set());t_stats=_fmt_stats(s);_update_stats_surfaces(font);auto.updated=False
				except Exception as e4:print('[FATAL] auto.updated apply failed:',e4,file=sys.stderr);traceback.print_exc();_cleanup_and_exit()
			try:
				use_dirty=not _FORCE_FULL_REPAINT and bool(_DIRTY_CELLS);dibujar_tablero(screen,show_player=not auto_mode,dirty_only=use_dirty);_draw_highlights(screen,time.monotonic_ns())
				if not auto_mode and menu.esta_abierto():menu.dibujar(screen)
				draw_ui(screen);draw_panel(screen);pygame.draw.rect(screen,STATS_BG,(WINDOW_SIZE,P1H,STATS_PANEL_W,WINDOW_SIZE));pygame.draw.line(screen,COL_SEP,(WINDOW_SIZE,P1H),(WINDOW_SIZE,P1H+WINDOW_SIZE));tx,ty=WINDOW_SIZE+STATS_PANEL_PADDING,P1H+STATS_PANEL_PADDING
				for surf in t_stats_surfaces:screen.blit(surf,(tx,ty));ty+=surf.get_height()+2
				pygame.display.flip();_FORCE_FULL_REPAINT=False
			except Exception as e5:print('[FATAL] draw phase failed:',e5,file=sys.stderr);traceback.print_exc();_cleanup_and_exit()
	except KeyboardInterrupt:print('[INTERRUPT] KeyboardInterrupt capturado. Cerrando...',file=sys.stderr);_cleanup_and_exit()
	except Exception as e:print('[FATAL] Unhandled exception en main loop:',e,file=sys.stderr);traceback.print_exc();_cleanup_and_exit()
class Game:
	def bucle_principal(self):main()
if __name__=='__main__':main()