import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # Carga el entorno usando el ID registrado
    env = gym.make('MiJuego-v0')

    # Crea el modelo de PPO con una política MLP
    model = PPO("MlpPolicy", env, verbose=1)

    # Entrena el modelo
    model.learn(total_timesteps=50000)

    # Guarda el modelo entrenado
    model.save("ppo_mi_juego")

    # Carga el modelo
    model = PPO.load("ppo_mi_juego")

    # Demostración del modelo
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
