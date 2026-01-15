# mapa_calor.py
import jax.numpy as jnp
import numpy as np
from config import BOARD_SIZE

class AnalistaTerreno:
    """
    Cerebro táctico 'Stockfish'.
    Evalúa el tablero completo en paralelo usando JAX.
    """
    def __init__(self):
        self.ultimas_coords_lote = None
        self.mascara_lote = None
        # Pre-calculamos la mascara de borde global (es estatica)
        # 1 = Construible, 0 = Lava (Borde Global)
        self.mascara_bordes_globales = self._crear_mascara_borde_global()

    def _normalizar(self, matriz):
        """Normaliza una matriz entre 0.0 y 1.0"""
        min_val = jnp.min(matriz)
        max_val = jnp.max(matriz)
        return jnp.where(max_val > min_val, (matriz - min_val) / (max_val - min_val), 0.0)

    def _crear_mascara_lote(self, coords):
        x0, y0, x1, y1 = coords
        mask = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
        mask = mask.at[x0:x1, y0:y1].set(1.0)
        return mask

    def _crear_mascara_borde_global(self):
        """
        Crea una mascara que anula estrictamente los bordes del mapa 200x200.
        Esto informa a la IA que esos lugares son prohibidos antes de intentar ir.
        """
        mask = jnp.ones((BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
        # Anulamos filas superior e inferior
        mask = mask.at[0, :].set(0.0)
        mask = mask.at[BOARD_SIZE-1, :].set(0.0)
        # Anulamos columnas izquierda y derecha
        mask = mask.at[:, 0].set(0.0)
        mask = mask.at[:, BOARD_SIZE-1].set(0.0)
        return mask

    def generar_mapa_tactico(self, target_building, occupancy, jax_maps, chunk_coords):
        """
        Genera un Mapa de Calor especifico para el edificio que queremos construir.
        """
        if occupancy.ndim == 3:
            ocupado = occupancy[:, :, 0]
        else:
            ocupado = occupancy
        
        espacio_libre = 1.0 - ocupado

        # Actualizamos mascara de lote si cambio el chunk
        if self.ultimas_coords_lote != chunk_coords:
            self.mascara_lote = self._crear_mascara_lote(chunk_coords)
            self.ultimas_coords_lote = chunk_coords
        
        # --- FILTRO MAESTRO DE LEGALIDAD ---
        # 1. Tiene que estar vacio.
        # 2. Tiene que estar en mi Lote actual.
        # 3. NO puede estar en el borde global (x=0,199 y=0,199).
        base_legal = espacio_libre * self.mascara_lote * self.mascara_bordes_globales

        # Recuperamos capas de informacion
        capa_agua = self._normalizar(jax_maps.get('agua', jnp.zeros_like(ocupado)))
        capa_energia = self._normalizar(jax_maps.get('energia', jnp.zeros_like(ocupado)))
        capa_felicidad = self._normalizar(jax_maps.get('felicidad', jnp.zeros_like(ocupado)))
        capa_seguridad = self._normalizar(jax_maps.get('seguridad', jnp.zeros_like(ocupado)))
        
        deseo = jnp.zeros_like(ocupado)

        # --- LOGICA DE DESEO POR TIPO ---
        if target_building == 'residencia':
            # Queremos: Agua, Energia, Felicidad. Seguridad es bono.
            score = (capa_agua * 2.0) + (capa_energia * 2.0) + (capa_felicidad * 1.0) + (capa_seguridad * 0.5)
            deseo = score * base_legal

        elif target_building in ('agua', 'lecheria', 'refineria', 'depuradora'):
            # Servicios basicos: Donde falten
            cobertura = jnp.zeros_like(ocupado)
            if target_building == 'agua': cobertura = capa_agua
            elif target_building == 'refineria': cobertura = capa_energia
            necesidad = 1.0 - cobertura
            deseo = necesidad * base_legal

        elif target_building in ('taller_togas', 'herreria'):
            # Industria: Donde haya energia
            deseo = (capa_energia * 1.0) * base_legal

        elif target_building in ('policia', 'bombero', 'hospital', 'colegio'):
            # Servicios rango: Donde falte seguridad/cobertura
            falta_seguridad = 1.0 - capa_seguridad
            deseo = falta_seguridad * base_legal
            
        else:
            # Default (Suelo, Decoracion): Solo importa que sea legal
            deseo = base_legal

        return deseo

    @staticmethod
    def convertir_para_visual(jax_map):
        try:
            return np.array(jax_map)
        except:
            return None