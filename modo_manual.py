# modo_manual.py
from __future__ import annotations
import pygame,uuid,time
from threading import Lock
from typing import Tuple,Dict,Any,Optional
from config import BOARD_SIZE
import edificios
from ml_replay import StepWriter
# IMPORT CORREGIDO: Usamos las funciones nuevas de JAX
from modo_automatico import _obs_fast, _rew_fast, IDX, BN_TO_ID, ACTION_SET
import scorer_jax as scorer

EVT_CITY_CHANGED=pygame.USEREVENT+1
EVT_HUD=pygame.USEREVENT+2
Coord=Tuple[int,int]

class ModoManual:
    __slots__='board','player_pos','speed','ed','sizes','S','updated','lock','episode_id','writer'
    def __init__(self,board,player_pos,lock=None):
        self.board=board;self.player_pos=player_pos;self.speed=2;self.ed=edificios.edificios
        self.sizes={k:tuple(map(int,v.tamanio))for(k,v)in self.ed.items()}
        self.S=int(BOARD_SIZE);self.updated=False;self.lock=lock or Lock()
        self.episode_id=int(time.time()*1000)
        self.writer=StepWriter(episode_id=self.episode_id,version=3,flush_every=5)
        print(f"[MANUAL] Grabando partida ID: {self.episode_id}")

    def _record_action(self,bn,x,y,ok):
        try:
            if bn not in IDX:return
            # 1. Reconstruir estado JAX al vuelo para la foto (Snapshot)
            jax_maps = scorer.make_maps()
            cnt = {k:0 for k in ACTION_SET if k!='demoler'}
            
            # Recorremos el tablero para llenar los mapas matematicos
            for (bx,by), val in self.board.items():
                bname = val[0] if isinstance(val, tuple) else str(val)
                if bname in BN_TO_ID:
                    jax_maps = scorer.apply_kernel_diff(jax_maps, bname, bx, by, 1.0)
                    cnt[bname] = cnt.get(bname, 0) + 1
            
            # 2. Generar Observacion compatible con la IA
            obs = _obs_fast(jax_maps, cnt)
            
            # 3. Calcular metrica
            ec_proxy = _rew_fast(jax_maps)
            r = 1.0 if ok else -0.1
            u,v = x/float(self.S), y/float(self.S)
            
            # 4. Guardar
            self.writer.append(int(ok),IDX[bn],float(u),float(v),float(r),float(ec_proxy),obs)
            self.writer.flush()
        except Exception as e:
            print(f"Error grabando manual: {e}")

    @staticmethod
    def _post(evt_type,payload=None):
        try:
            if pygame.get_init():pygame.event.post(pygame.event.Event(evt_type,payload or{}))
        except Exception:pass

    def _post_city_changed(self):self._post(EVT_CITY_CHANGED,None)
    def _emit_hud(self,kind,bn,x,y,w,h):self._post(EVT_HUD,{'kind':kind,'bn':bn,'x':x,'y':y,'w':w,'h':h,'hv':None})
    @staticmethod
    def _new_iid_u64():return uuid.uuid4().int&(1<<64)-1

    def update_movement(self):k=pygame.key.get_pressed();dx=k[pygame.K_RIGHT]-k[pygame.K_LEFT];dy=k[pygame.K_DOWN]-k[pygame.K_UP];x,y=self.player_pos;s,M=self.speed,self.S;self.player_pos[0]=max(0,min(M-1,int(x+dx*s)));self.player_pos[1]=max(0,min(M-1,int(y+dy*s)))

    def _bloque(self,x,y,w,h):
        if x<0 or y<0 or x+w>self.S or y+h>self.S:return[]
        return[(x+i,y+j)for i in range(w)for j in range(h)]

    def _act_build(self,selected_building):
        bn=selected_building
        if not bn:print('Debes seleccionar un edificio');return selected_building
        if bn not in self.sizes:print('Edificio desconocido');return selected_building
        w,h=self.sizes.get(bn,(1,1));px,py=int(self.player_pos[0]),int(self.player_pos[1]);x=min(px,self.S-w);y=min(py,self.S-h)
        with self.lock:
            coords=self._bloque(x,y,w,h)
            if not coords:print('Fuera de límites');self._record_action(bn,x,y,False);return selected_building
            if any(c in self.board for c in coords):print('Hay un edificio en la casilla');self._record_action(bn,x,y,False);return selected_building
            is_single_str=bn in('suelo','decoracion')and w==1 and h==1
            def _place_single():self.board[x,y]=bn
            def _place_multi():iid=self._new_iid_u64();self.board.update({c:(bn,iid)for c in coords})
            {True:_place_single,False:_place_multi}[is_single_str]()
        try:print(f"Edificio {bn} construido en {coords[:3]}{'...' if len(coords)>3 else ''}")
        except Exception:pass
        self.updated=True;self._emit_hud('placed',bn,x,y,w,h);self._post_city_changed()
        self._record_action(bn,x,y,True)
        return selected_building

    def _act_demolish(self,selected_building):
        px,py=int(self.player_pos[0]),int(self.player_pos[1]);pos=px,py
        with self.lock:
            v=self.board.get(pos)
            if v is None:print('No hay edificio en la casilla');self._record_action('demoler',px,py,False);return selected_building
            def _dem_tuple():
                bn,iid=v;minx=miny=10**9;maxx=maxy=-1
                for(c,val)in list(self.board.items()):
                    if val==(bn,iid):
                        x,y=c
                        if x<minx:minx=x
                        if y<miny:miny=y
                        if x>maxx:maxx=x
                        if y>maxy:maxy=y
                        del self.board[c]
                return bn,max(minx,0),max(miny,0),maxx-minx+1,maxy-miny+1
            def _dem_scalar():bn=str(v);del self.board[pos];return bn,px,py,1,1
            bn,rx,ry,rw,rh={tuple:_dem_tuple,str:_dem_scalar}.get(type(v),_dem_scalar)()
        try:print(f"Edificio {bn} demolido")
        except Exception:pass
        self.updated=True;self._emit_hud('demolished',bn,rx,ry,rw,rh);self._post_city_changed()
        self._record_action('demoler',rx,ry,True)
        return selected_building

    def _act_show_coords(self,selected_building):print(f"Coordenadas del jugador: ({int(self.player_pos[0])}, {int(self.player_pos[1])})");return selected_building
    
    def _act_open_menu(self,menu,selected_building):
        try:menu.abrir()
        except Exception:pass
        return selected_building

    def handle_event(self,event,menu,selected_building):k,u=event.key,event.unicode;key_handlers={pygame.K_PLUS:lambda:self._act_build(selected_building),pygame.K_KP_PLUS:lambda:self._act_build(selected_building),pygame.K_MINUS:lambda:self._act_demolish(selected_building),pygame.K_KP_MINUS:lambda:self._act_demolish(selected_building),pygame.K_n:lambda:self._act_show_coords(selected_building),pygame.K_m:lambda:self._act_open_menu(menu,selected_building)};uni_handlers={'+':lambda:self._act_build(selected_building),'-':lambda:self._act_demolish(selected_building)};handler=key_handlers.get(k)or uni_handlers.get(u)or(lambda:selected_building);return handler()