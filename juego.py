# juego.py
import pygame,sys,signal,threading,time,traceback
import capnp, io, os
import numpy as np 
from collections import deque
from config import BOARD_SIZE,WINDOW_SIZE,CELL_SIZE,BACKGROUND_COLOR,PLAYER_COLOR,PANEL_TOP_H,PANEL_BOTTOM_H,STATS_PANEL_W,STATS_PANEL_PADDING
from menu import Menu
from modo_manual import ModoManual
from modo_automatico import ModoAutomatico
from estadisticas import StatsProcessor
import edificios,bd_celdas
import restricciones

cc = capnp.load('celdas.capnp')

_shutdown_event=threading.Event()
def _signal_handler(sig,frame):
    _shutdown_event.set()
    try:
        if pygame.get_init():pygame.event.post(pygame.event.Event(pygame.QUIT))
    except Exception:pass
signal.signal(signal.SIGINT,_signal_handler)
try:signal.signal(signal.SIGTERM,_signal_handler)
except:pass

EVT_CITY_CHANGED=pygame.USEREVENT+1
EVT_HUD=pygame.USEREVENT+2
P1H=PANEL_TOP_H; P3H=PANEL_BOTTOM_H
# Config de Botones
BXS=20; BSZ=128,36
BTN_GAP=10

COL_P1=BACKGROUND_COLOR; COL_P3=BACKGROUND_COLOR
COL_SEP=60,60,60; COL_TEXT=230,230,230
COL_BTN=55,55,55; COL_BH=75,75,75; COL_BB=90,90,90
COL_ACTIVE=76,194,255; STATS_BG=BACKGROUND_COLOR

# --- COLORES DE ESTADO ---
COL_DIM=(70, 70, 70) 
COL_INACTIVE=(255, 255, 255) # BLANCO (Para que se vean normales si no hay filtro)

ALLOW_STATIC={'suelo','decoracion'}

LOCK_PROVIDERS={'energia':'refineria','agua':'agua','comida':'lecheria','basura':'depuradora'}

t_stats=[]
t_stats_surfaces=[]
board={}
building_counts={} # ALMACEN PARA EL CENSO
_BOARD_LOCK=None
player_pos=[BOARD_SIZE//2,BOARD_SIZE//2]
_last_player_pos=tuple(player_pos)
selected_building=None
auto_mode=False
show_heat_vision = False 
current_heatmap = None   
lock_resources_ui=set()
HIGHLIGHTS=deque(maxlen=64)
HIGHLIGHT_MS=950
NS_PER_MS=1000000
_DIRTY_CELLS=set()
_FORCE_FULL_REPAINT=True
inactive_ids_from_ai=set()

# --- DEFINICION DE BOTONES ---
def _get_btn_auto_rect():
    return pygame.Rect(BXS, (P1H-BSZ[1])//2, *BSZ)

def _get_btn_heat_rect():
    r1 = _get_btn_auto_rect()
    return pygame.Rect(r1.right + BTN_GAP, (P1H-BSZ[1])//2, *BSZ)

def _u64(h):
    try:return int(h)&0xffffffffffffffff
    except:return 0
def _is_anchor(h):h=_u64(h);return bool(h)and h&255==0

def _calc_lock_resources(s):
    if not s: return set()
    return {k for k,v in {
        'energia': s.energiaTotal+s.energiaUsada,
        'agua': s.aguaTotal+s.aguaUsada,
        'comida': s.comidaTotal+s.comidaUsada,
        'basura': s.basuraTotal+s.basuraUsada
    }.items() if v < 0}

def _realizar_censo():
    """Cuenta cuantos edificios de cada tipo hay en el tablero actualmente"""
    global building_counts
    temp_counts = {}
    lk = globals().get('_BOARD_LOCK')
    
    # Leemos de manera segura
    snapshot = []
    if lk:
        with lk: snapshot = list(board.values())
    else:
        snapshot = list(board.values())

    for val in snapshot:
        nombre = val[0] if isinstance(val, tuple) else val
        temp_counts[nombre] = temp_counts.get(nombre, 0) + 1
    
    building_counts = temp_counts

def _recalc_ui_stats():
    board_snapshot = None
    lk = globals().get('_BOARD_LOCK')
    try:
        if lk:
            with lk: board_snapshot = board.copy()
        else:
            board_snapshot = board.copy()
            
        s1 = StatsProcessor().process_ram(board_snapshot, inactive_set=inactive_ids_from_ai)
        
        lr = _calc_lock_resources(s1)
        return lr, s1
    except Exception as e:
        print(f"[UI] Error Stats: {e}")
        return set(), None

def _fmt_stats(s):
    if not s: return ["Cargando..."]
    txt = [f"Residentes: {s.totalResidentes}",f"Empleos: {s.totalEmpleos}",
           f"  Industria: {s.totalEmpleosIndustria} ({s.porcentajeIndustria:.1f}%)",
           f"  Comercio: {s.totalEmpleosComercio} ({s.porcentajeComercio:.1f}%)",
           f"Desbalance: {s.desequilibrioLaboral}",
           f"Energía R/T: {s.energiaUsada}/{s.energiaTotal}",
           f"Agua R/T: {s.aguaUsada}/{s.aguaTotal}",
           f"Comida R/T: {s.comidaUsada}/{s.comidaTotal}",
           f"Basura R/T: {s.basuraUsada}/{s.basuraTotal}",
           f"Ecología: {s.ecologiaTotal}",f"Felicidad: {s.felicidadTotal}"]
    
    # AÑADIMOS EL CENSO AL FINAL DEL TEXTO
    txt.append("")
    txt.append("--- CENSO (5s) ---")
    if building_counts:
        # Ordenamos alfabéticamente o por cantidad
        for nombre, cantidad in sorted(building_counts.items(), key=lambda x: x[1], reverse=True):
            txt.append(f"{nombre.capitalize()}: {cantidad}")
    else:
        txt.append("Calculando...")

    return txt

def _update_stats_surfaces(font):
    global t_stats_surfaces
    t_stats_surfaces=[font.render(line,True,COL_TEXT)for line in t_stats]

def _cell_rect(x,y):cs,y0=CELL_SIZE,PANEL_TOP_H;return pygame.Rect(x*cs,y*cs+y0,cs,cs)
def _mark_dirty_rect(x,y,w,h):
    for i in range(w):
        for j in range(h):_DIRTY_CELLS.add((x+i,y+j))

def _snapshot_all_items():
    lk=globals().get('_BOARD_LOCK')
    if lk:
        with lk:return tuple(board.items())
    return tuple(board.items())

def dibujar_tablero(p,show_player,*,dirty_only):
    cs,y0=CELL_SIZE,P1H;e=edificios.edificios;rect=pygame.draw.rect
    allow={LOCK_PROVIDERS[r]for r in lock_resources_ui if r in LOCK_PROVIDERS}if lock_resources_ui else None
    
    if not dirty_only or auto_mode:
        p.fill(BACKGROUND_COLOR)
        for((x,y),b)in _snapshot_all_items():
            name=b[0]if isinstance(b,tuple)else b
            
            is_active_vis = True
            if not auto_mode and name not in ('suelo', 'decoracion'):
                try:
                    w,h = 1,1
                    if name in e: w,h = map(int, e[name].tamanio)
                    bloque = set((x+i, y+j) for i in range(w) for j in range(h))
                    is_active_vis = restricciones.es_activo(board, name, bloque)
                except: pass
            
            is_active_ai = True
            if auto_mode:
                if isinstance(b, tuple): 
                    if b[1] in inactive_ids_from_ai: is_active_ai = False
                else: 
                    if (x,y) in inactive_ids_from_ai: is_active_ai = False

            is_dimmed = not (is_active_ai if auto_mode else is_active_vis)

            if allow and name.lower() not in allow and name.lower() not in ALLOW_STATIC:
                is_dimmed = True

            if is_dimmed: col = COL_INACTIVE
            elif name in e: col = e[name].color
            else: col = COL_DIM

            rect(p,col,(x*cs,y*cs+y0,cs,cs))
    else:
        wanted=set(_DIRTY_CELLS)
        if not wanted:return
        for(x,y)in wanted:
            b=board.get((x,y))
            rect(p,BACKGROUND_COLOR,(x*cs,y*cs+y0,cs,cs))
            if b is not None:
                name=b[0]if isinstance(b,tuple)else b
                is_active = True
                if name not in ('suelo', 'decoracion'):
                    try:
                        w,h = 1,1
                        if name in e: w,h = map(int, e[name].tamanio)
                        bloque = set((x+i, y+j) for i in range(w) for j in range(h))
                        is_active = restricciones.es_activo(board, name, bloque)
                    except: pass

                is_dimmed = not is_active
                if allow and name.lower() not in allow and name.lower() not in ALLOW_STATIC:
                    is_dimmed = True

                if is_dimmed: col = COL_INACTIVE
                elif name in e: col = e[name].color
                else: col = COL_DIM

                rect(p,col,(x*cs,y*cs+y0,cs,cs))
        _DIRTY_CELLS.clear()
    
    if show_heat_vision and current_heatmap is not None:
        overlay = pygame.Surface((BOARD_SIZE*CELL_SIZE, BOARD_SIZE*CELL_SIZE), pygame.SRCALPHA)
        rows, cols = current_heatmap.shape
        for r in range(rows):
            for c in range(cols):
                val = current_heatmap[r, c]
                if val > 0.1: 
                    alpha = int(min(val * 180, 150)) 
                    color = (0, 255, 0, alpha) 
                    pygame.draw.rect(overlay, color, (r*CELL_SIZE, c*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        p.blit(overlay, (0, P1H))

    if show_player:rect(p,PLAYER_COLOR,(player_pos[0]*cs,player_pos[1]*cs+y0,cs,cs))

def _draw_highlights(p,now_ns):
    if not HIGHLIGHTS:return
    cs,y0=CELL_SIZE,P1H;keep=[];draw_rect=pygame.draw.rect
    for(bn,x,y,w,h,t_end_ns,kind,hv)in list(HIGHLIGHTS):
        if now_ns>t_end_ns:continue
        col=(220,80,80)if kind=='demolished'else COL_ACTIVE
        draw_rect(p,col,pygame.Rect(x*cs,y*cs+y0,w*cs,h*cs),width=2)
        keep.append((bn,x,y,w,h,t_end_ns,kind,hv))
    HIGHLIGHTS.clear();HIGHLIGHTS.extend(keep)

def _draw_generic_btn(p, rect, text, is_active, is_enabled=True):
    hov = is_enabled and rect.collidepoint(pygame.mouse.get_pos())
    base = COL_BH if hov else COL_BTN
    pygame.draw.rect(p, base, rect, border_radius=4)
    pygame.draw.rect(p, COL_BB, rect, 1, border_radius=4)
    
    color_txt = (160,160,160) if not is_enabled else COL_ACTIVE if is_active else COL_TEXT
    t = pygame.font.SysFont(None, 18).render(text, True, color_txt)
    p.blit(t, (rect.x + (rect.width - t.get_width())//2, rect.y + (rect.height - t.get_height())//2))

def dibujar_ui(p,boton_habilitado=True):
    pygame.draw.rect(p,COL_P1,(0,0,WINDOW_SIZE,P1H))
    pygame.draw.line(p,COL_SEP,(0,P1H-1),(WINDOW_SIZE,P1H-1))
    
    _draw_generic_btn(p, _get_btn_auto_rect(), "Auto Mode", auto_mode, boton_habilitado)
    
    has_data = current_heatmap is not None
    _draw_generic_btn(p, _get_btn_heat_rect(), "Visión Calor", show_heat_vision, is_enabled=has_data)

    lbl=pygame.font.SysFont(None,18).render(f"Edif seleccionado: {selected_building or 'Ninguno'}",True,COL_TEXT)
    p.blit(lbl,(_get_btn_heat_rect().right+20,(P1H-lbl.get_height())//2))
    
    title=pygame.font.SysFont(None,18).render('Estadísticas',True,COL_TEXT)
    p.blit(title,(WINDOW_SIZE+STATS_PANEL_PADDING,(P1H-title.get_height())//2))

def dibujar_panel_fino(p):pygame.draw.rect(p,COL_P3,(0,WINDOW_SIZE+P1H,WINDOW_SIZE,P3H))

def _set_auto_mode(enable,menu):
    global auto_mode,_FORCE_FULL_REPAINT
    auto_mode=enable;pygame.mouse.set_visible(True);_FORCE_FULL_REPAINT=True
    fn=getattr(menu,'set_enabled',None)
    try:fn and fn(not enable)
    except:pass
    if enable and getattr(menu,'cerrar',None):
        try:menu.cerrar()
        except:pass

def _cargar_tablero_inicial():
    global board
    try:
        with open("celdas.bin", "rb") as f:
            mapa = cc.Mapa.read(f)
            for c in mapa.celdas:
                bn = c.edificio
                if bn:
                    h = int(c.hash) if c.hash else 0
                    iid = (h >> 8) & 0xffffffffffffffff
                    if iid > 0: board[(c.x, c.y)] = (str(bn), str(iid))
                    else: board[(c.x, c.y)] = str(bn)
        print(f"[INIT] Tablero cargado. Elementos: {len(board)}")
    except FileNotFoundError:
        print("[INIT] Sin celdas.bin. Tablero vacio.")
    except Exception as e:
        print(f"[INIT] Archivo corrupto detectado ({e}). Eliminando y reiniciando.")
        board = {}
        try:
            os.remove("celdas.bin")
            print("[INIT] celdas.bin eliminado exitosamente.")
        except: pass

def main():
    global t_stats,t_stats_surfaces,selected_building,lock_resources_ui,_BOARD_LOCK,_FORCE_FULL_REPAINT,_last_player_pos, inactive_ids_from_ai, show_heat_vision, current_heatmap
    
    _cargar_tablero_inicial()
    
    # Hacemos el primer censo al arrancar
    _realizar_censo()

    pygame.init();flags=pygame.HWSURFACE|pygame.DOUBLEBUF
    w,h=WINDOW_SIZE+STATS_PANEL_W,WINDOW_SIZE+P1H+P3H
    try:screen=pygame.display.set_mode((w,h),flags,vsync=1)
    except TypeError:screen=pygame.display.set_mode((w,h),flags)
    pygame.display.set_caption('DEG - CityBuilder')
    clock=pygame.time.Clock();font=pygame.font.SysFont(None,16)
    pygame.mouse.set_visible(True)
    pygame.event.set_allowed([pygame.QUIT,pygame.KEYDOWN,pygame.MOUSEBUTTONDOWN,EVT_HUD,EVT_CITY_CHANGED])
    ev_get=pygame.event.get;monotonic_ns=time.monotonic_ns
    draw_tab=dibujar_tablero;draw_ui=dibujar_ui;draw_panel=dibujar_panel_fino
    
    lock=threading.Lock();_BOARD_LOCK=lock
    menu=Menu()
    manual=ModoManual(board,player_pos,lock=lock)
    auto=ModoAutomatico(board,lock=lock)
    
    def _cleanup_and_exit():
        try:pygame.quit()
        except:pass
        sys.exit(0)

    try:
        lr,s=_recalc_ui_stats()
        t_stats=_fmt_stats(s);_update_stats_surfaces(font)
    except:pass

    def _on_quit(e):_cleanup_and_exit()
    def _on_hud(e):
        ed=getattr(e,'dict',e.__dict__);bn=ed.get('bn','?');x=int(ed.get('x',0));y=int(ed.get('y',0))
        w_=int(ed.get('w')or 1);h_=int(ed.get('h')or 1);kind=ed.get('kind','placed')
        now_ns=monotonic_ns()
        HIGHLIGHTS.append((bn,x,y,w_,h_,now_ns+HIGHLIGHT_MS*NS_PER_MS,kind,None))
        _mark_dirty_rect(x,y,w_,h_)

    def _on_city_changed(e):
        global lock_resources_ui,t_stats,_FORCE_FULL_REPAINT
        if not auto_mode:
            try:
                with _BOARD_LOCK:
                    bd_celdas.update_celdas_bin(board, lock_resources=lock_resources_ui)
                lr,s = _recalc_ui_stats()
                lock_resources_ui=lr;t_stats=_fmt_stats(s);_update_stats_surfaces(font)
                _FORCE_FULL_REPAINT=True;manual.updated=False
            except Exception as e:
                print(f"[MANUAL SAVE ERROR] {e}")

    def _toggle_auto():_set_auto_mode(not auto_mode,menu)
    def _toggle_heat(): 
        global show_heat_vision, _FORCE_FULL_REPAINT
        show_heat_vision = not show_heat_vision
        _FORCE_FULL_REPAINT = True

    def ia_sigue_trabajando():
        return auto.running or (auto.thread and auto.thread.is_alive())

    def _key_default(e):
        global selected_building
        if ia_sigue_trabajando(): return

        if auto_mode:return
        if menu.esta_abierto():
            op=menu.manejar_evento(e)
            if op:selected_building=op
        else:
            try:
                with _BOARD_LOCK:
                    sel=manual.handle_event(e,menu,selected_building)
                if sel!=selected_building:selected_building=sel
            except Exception as e:
                print(f"[MANUAL ERR] {e}")

    key_handlers={
        pygame.K_a: _toggle_auto,
        pygame.K_h: _toggle_heat
    }
    def _on_keydown(e):(key_handlers.get(e.key)or _key_default)(e)
    
    def _on_mouse(e):
        if _get_btn_auto_rect().collidepoint(e.pos):
            _set_auto_mode(not auto_mode, menu)
        elif _get_btn_heat_rect().collidepoint(e.pos):
            _toggle_heat()

    def _noop(e):0
    
    handlers={pygame.QUIT:_on_quit,EVT_HUD:_on_hud,EVT_CITY_CHANGED:_on_city_changed,pygame.KEYDOWN:_on_keydown,pygame.MOUSEBUTTONDOWN:_on_mouse}
    last_auto_stat_time=0
    
    # NUEVO: Timer para el censo de 5 segundos
    last_census_time = 0
    
    try:
        while True:
            if _shutdown_event.is_set():_cleanup_and_exit()
            clock.tick(60)
            now_for_census = time.time()
            
            # --- LOGICA DEL CENSO CADA 5 SEGUNDOS ---
            if now_for_census - last_census_time > 5.0:
                _realizar_censo()
                last_census_time = now_for_census
                # Forzamos actualizacion de texto
                lr,s=_recalc_ui_stats()
                if s:
                    t_stats=_fmt_stats(s);_update_stats_surfaces(font)
                    lock_resources_ui=lr
            # ----------------------------------------

            if _last_player_pos!=tuple(player_pos):
                ox,oy=_last_player_pos;_mark_dirty_rect(ox,oy,1,1)
                nx,ny=player_pos;_mark_dirty_rect(nx,ny,1,1)
                _last_player_pos=tuple(player_pos)

            for e in ev_get():
                try:(handlers.get(e.type)or _noop)(e)
                except Exception as ex_loop:
                    print(f"[LOOP] Error evento: {ex_loop}")

            auto.actualizar(1 if auto_mode else 0, 0)

            if not auto.running and getattr(auto, 'thread', None) and not auto.thread.is_alive() and not auto_mode:
                 pass 

            if not auto_mode and not menu.esta_abierto() and not ia_sigue_trabajando():
                manual.update_movement()
            
            if auto_mode:
                now=time.time()
                if now-last_auto_stat_time > 0.5:
                    inactive_ids_from_ai = auto.get_inactive_ids()
                    current_heatmap = auto.get_heatmap() 
                    
                    lr,s=_recalc_ui_stats()
                    if s:
                        t_stats=_fmt_stats(s);_update_stats_surfaces(font)
                        lock_resources_ui=lr
                        last_auto_stat_time=now
                        _FORCE_FULL_REPAINT=True

            try:
                use_dirty=not _FORCE_FULL_REPAINT and bool(_DIRTY_CELLS) and not auto_mode and not show_heat_vision
                draw_tab(screen,show_player=not auto_mode,dirty_only=use_dirty)
                _draw_highlights(screen,time.monotonic_ns())
                if not auto_mode and menu.esta_abierto():menu.dibujar(screen)
                draw_ui(screen);draw_panel(screen)
                
                pygame.draw.rect(screen,STATS_BG,(WINDOW_SIZE,P1H,STATS_PANEL_W,WINDOW_SIZE))
                pygame.draw.line(screen,COL_SEP,(WINDOW_SIZE,P1H),(WINDOW_SIZE,P1H+WINDOW_SIZE))
                tx,ty=WINDOW_SIZE+STATS_PANEL_PADDING,P1H+STATS_PANEL_PADDING
                for surf in t_stats_surfaces:
                    screen.blit(surf,(tx,ty));ty+=surf.get_height()+2
                
                pygame.display.flip()
                _FORCE_FULL_REPAINT=False
            except Exception as e5:print(f"Draw Err: {e5}");_cleanup_and_exit()

    except KeyboardInterrupt:_cleanup_and_exit()
    except Exception as e:print(e);_cleanup_and_exit()

class Game:
    def bucle_principal(self):main()

if __name__=='__main__':main()