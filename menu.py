# menu.py
import pygame, edificios
from config import WINDOW_SIZE, PANEL_TOP_H

class Menu:
    __slots__ = (
        'op', 'len', 'idx', 'o', 'enabled', 'f',
        'mw', 'mh', 'mx', 'my', 'sf', 'ssf',
        'bg', 'bg_shadow', 'border'
    )

    def __init__(self):
        ops = list(edificios.edificios.keys())
        ops.append("Planificar ciudad perfecta")  # ← nueva opción
        self.op = ops
        self.len = len(ops)
        self.idx = 0
        self.o = False
        self.enabled = True
        f = pygame.font.SysFont("Arial", 20)
        self.f = f
        self.mw = 300
        self.mh = 20 + self.len * 30
        self.mx = (WINDOW_SIZE - self.mw) // 2
        self.my = PANEL_TOP_H + ((WINDOW_SIZE - self.mh) // 2)
        rf = f.render
        self.sf = [rf(u, True, (200, 200, 200)) for u in ops]
        self.ssf = [rf(u, True, (255, 215, 0)) for u in ops]
        self.bg = pygame.Rect(self.mx, self.my, self.mw, self.mh)
        self.bg_shadow = pygame.Rect(self.mx + 4, self.my + 4, self.mw, self.mh)
        self.border = self.bg

    # control
    def set_enabled(self, v: bool):
        self.enabled = bool(v)

    def abrir(self):
        self.o = self.enabled
        self.idx = 0

    def cerrar(self):
        self.o = False

    def esta_abierto(self):
        return self.o

    # draw
    def dibujar(self, s):
        if not (self.o and self.enabled):
            return
        dr = pygame.draw.rect
        mx, my, mw = self.mx, self.my, self.mw
        slf = self.sf
        sslf = self.ssf
        idx = self.idx
        dr(s, (20, 20, 20), self.bg_shadow, border_radius=8)
        dr(s, (40, 40, 40), self.bg, border_radius=8)
        dr(s, (80, 80, 80), self.border, 2, border_radius=8)
        y = my + 10
        blit = s.blit
        for i in range(self.len):
            blit(sslf[i] if i == idx else slf[i], (mx + 10, y))
            y += 30

    # input
    def manejar_evento(self, e):
        if not (self.enabled and self.o) or e.type != pygame.KEYDOWN:
            return None
        k = e.key
        n = self.len
        if k == pygame.K_UP:
            self.idx = (self.idx - 1) % n
        elif k == pygame.K_DOWN:
            self.idx = (self.idx + 1) % n
        elif k in (pygame.K_RETURN, pygame.K_m):
            sel = self.op[self.idx]
            self.cerrar()
            if sel == "Planificar ciudad perfecta":
                return "__PLAN_CIUDAD_PERFECTA__"  # código especial para manejar en juego.py
            return sel
        return None
