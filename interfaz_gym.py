import gymnasium as gym
import numpy as np
import pickle
from main import Stats, generar_hash, Menu, actualizar_pantalla, dibujar_cuadricula  # Asegúrate de importar funciones correctas
from main import TECLA_IZQUIERDA, TECLA_DERECHA, TECLA_ARRIBA, TECLA_ABAJO, ANCHO_VENTANA, ALTO_VENTANA, NUM_CELDAS
from main import edificios, mapa, generar_datos, generar_datos_stats  # Importa las variables y funciones necesarias del juego


class MiEntornoJuego(gym.Env):
    def __init__(self):
        super().__init__()

        # Inicializa el juego
        self.juego = Stats()  # Utiliza la clase Stats si gestiona el estado del juego

        # Define los espacios de observación y acción
        self.espacio_observacion = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.espacio_accion = gym.spaces.Discrete(4)

        # Estado inicial del juego
        self.estado = self.obtener_estado_inicial()

    def step(self, action):
        # Decodifica y aplica la acción
        action_decoded = self.decode_action(action)
        reward, done = self.apply_action(action_decoded)

        # Actualiza y obtiene la nueva observación
        observation = self.get_observation()
        return observation, reward, done, {}

    def reset(self):
        # Reinicia el juego y actualiza el estado
        self.juego.reset_game()  # Asegúrate de que esta función exista
        self.estado = self.obtener_estado_inicial()
        return self.get_observation()

    def obtener_estado_inicial(self):
        # Carga el estado inicial del juego
        with open('celdas.pkl', 'rb') as file:
            celdas = pickle.load(file)
        return celdas

    def decode_action(self, action):
        # Traduce acciones de entero a acciones en el juego
        directions = {0: "izquierda", 1: "derecha", 2: "arriba", 3: "abajo"}
        return directions.get(action, "acción desconocida")

    def apply_action(self, action):
        # Aplica la acción en el juego; deberías conectar esto con tu lógica de juego real
        if action == "izquierda":
            # Aplica la lógica correspondiente
            pass
        return 0, False  # Retorna la recompensa y si el juego terminó

    def get_observation(self):
        # Devuelve el estado actual del juego como observación
        with open('estadisticas.pkl', 'rb') as file:
            stats = pickle.load(file)
        return stats

# Registro del entorno en Gymnasium
from gymnasium.envs.registration import register

register(
    id='MiJuego-v0',
    entry_point='interfaz_gym:MiEntornoJuego'  # Asegúrate de que el nombre del módulo es correcto
)
