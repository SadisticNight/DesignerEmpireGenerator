import gymnasium as gym
from stable_baselines3 import PPO
import interfaz_gym
from custom_policy import CustomCNN  # Asegúrate de que este import está correcto

def main():
    # Inicializa el entorno
    env = gym.make('MiJuego-v0')

    # Configura el modelo PPO para usar la política personalizada CustomCNN
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256)  # Asegúrate de que esto corresponde con tu clase CustomCNN
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Entrena el modelo
    model.learn(total_timesteps=50000)

    # Guarda el modelo entrenado
    model.save("ppo_mi_juego")

    # Carga el modelo
    model = PPO.load("ppo_mi_juego")

    # Demostración de cómo el modelo juega el juego automáticamente
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()  # Asegúrate de que tu entorno soporte renderización
        if dones:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
