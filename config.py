# config.py
# Constantes mínimas y derivadas

# --- Tablero ---
BOARD_SIZE  = 200
WINDOW_SIZE = 600
assert WINDOW_SIZE % BOARD_SIZE == 0, "WINDOW_SIZE debe ser múltiplo de BOARD_SIZE"
CELL_SIZE   = WINDOW_SIZE // BOARD_SIZE
GRID_CELLS  = BOARD_SIZE * BOARD_SIZE

# --- Paneles UI ---
PANEL_TOP_H         = 50
PANEL_BOTTOM_H      = 8
STATS_PANEL_W       = 240
STATS_PANEL_PADDING = 12

# Dimensiones totales útiles (canvas + panel de stats)
CANVAS_W = WINDOW_SIZE + STATS_PANEL_W
CANVAS_H = WINDOW_SIZE + PANEL_TOP_H + PANEL_BOTTOM_H

# --- Colores ---
BACKGROUND_COLOR = (30, 30, 30)
PLAYER_COLOR     = (0, 255, 0)
STATS_PANEL_BG   = BACKGROUND_COLOR

# --- Flags de ejecución ---
AUTO_USE_SCORER  = True
AUTO_BATCH_EVERY = 1

# --- Logging / depuración ---
LOG_COLOCACIONES = True
LOG_FALLOS       = True

# --- Objetivos de proporciones (planner legacy) ---
OBJ_PCT_COMERCIO  = 67.0
OBJ_PCT_INDUSTRIA = 33.0

# --- Pesos del score (planner legacy) ---
PESO_DESBALANCE = 10.0
PESO_SERVICIOS  = 5.0
PESO_FELICIDAD  = 3.0
PESO_AMBIENTE   = 3.0
PESO_INACTIVOS  = 15.0
PESO_SUELO      = 1.0

# --- Parámetros del optimizador/planner (legacy) ---
PLANNER_STEPS = 1000
PLANNER_BATCH = 10
PLANNER_SEED  = None

# --- ML / IO / HUD ---
STEPWRITER_FLUSH_EVERY = 1
AUTOSAVE_EVERY   = 512
AUTOSYNC_EVERY   = 2048
HUD_HIGHLIGHT_MS = 950

# Cadencia del bucle del agente
ML_LOOP_SLEEP_MS = 4
AUTO_TICK_SLEEP  = ML_LOOP_SLEEP_MS / 1000.0

# Caches / servicios
SRV_CACHE_PERIOD = 64

# --- Shaping (servicios / bienestar) ---
SRV_OVERLAP_FREE_ORIGINS = 4
K_SRV_OVERLAP_SOFT = -0.15
K_SRV_OVERLAP_HARD = -0.80
K_ECO_DELTA = 0.03
K_FEL_DELTA = 0.03

# --- Intrínseco (RND) ---
RND_ENABLED = True          # si el caller lo usa
RND_BETA    = 0.10          # peso de r_int al sumar a la recompensa externa
RND_H1      = 128           # dims MLP predictor/target
RND_H2      = 128
RND_OUT     = 64
RND_LR      = 1e-3          # solo predictor
RND_EMA     = 0.99          # normalización de r_int

# --- Referencia rápida de variables de entorno soportadas ---
# Core loop:
#   DEG_ENV_SEED, DEG_TICK_SLEEP, DEG_FLUSH_EVERY, DEG_SRV_CACHE
# Shaping/servicios:
#   DEG_SRV_FREE_ORIGINS, DEG_K_SRV_OVERLAP_SOFT, DEG_K_SRV_OVERLAP_HARD
#   DEG_K_ECO_DELTA, DEG_K_FEL_DELTA
# Modo mantenimiento:
#   DEG_MAINTENANCE, DEG_MAINT_IDLE,
#   DEG_TARGET_COV_POLICIA, DEG_TARGET_COV_BOMBERO,
#   DEG_TARGET_COV_COLEGIO, DEG_TARGET_COV_HOSPITAL
# Cobertura por edificio:
#   DEG_BLD_COVER_MODE (any|centroid|all), DEG_K_BLD_COV_GAIN,
#   DEG_K_BLD_UNCOV, DEG_K_BLD_OVERLAP
# Intrínseco (RND):
#   DEG_RND_BETA, DEG_RND_H1, DEG_RND_H2, DEG_RND_OUT, DEG_RND_LR, DEG_RND_EMA
