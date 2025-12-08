import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

def make_env():
    # figure it out later
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(env, "./logs/")
        return env
    return _init


def main():
    env = DummyVecEnv([make_env()])

    eval_env = DummyVecEnv([make_env()])

    eval_callback = EvalCallback(
            eval_env,
            eval_freq=10_000,
            n_eval_episodes=5,
            best_model_save_path="./models/",
            log_path="./logs/",
            deterministic=True,
            render=False,
            verbose=1,
            )

    model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log="./tb_logs/"
            )

    total_timesteps = 200_000

    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        print("\nInterrupt, saving current model")
        model.save("models/ppo_bipedalwalker_interrupt")
    else:
        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_bipedalwalker_final")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
