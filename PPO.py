import os
import time
import csv

import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback


class EpisodeStatsCallback(BaseCallback):

    def __init__(self, log_dir: str, window_size: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.window_size = window_size

        os.makedirs(self.log_dir, exist_ok=True)

        self.per_episode_path = os.path.join(self.log_dir, "episode_stats_per_episode.csv")
        self.avg_path = os.path.join(
            self.log_dir, f"episode_stats_avg_{self.window_size}.csv"
        )

      
        self.episode_count = 0
        self.window_rewards = []
        self.window_lengths = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()

        
        if not os.path.isfile(self.per_episode_path):
            with open(self.per_episode_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["episode", "total_timesteps", "reward", "length", "time_elapsed_sec"]
                )

        if not os.path.isfile(self.avg_path):
            with open(self.avg_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "episode_end",          
                        "total_timesteps",     
                        f"avg_reward_{self.window_size}",
                        f"avg_length_{self.window_size}",
                        "time_elapsed_sec",
                    ]
                )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                reward = float(ep_info["r"])
                length = int(ep_info["l"])

                self.episode_count += 1
                time_elapsed = time.time() - self.start_time
                total_timesteps = self.num_timesteps 

            
                with open(self.per_episode_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            self.episode_count,
                            total_timesteps,
                            reward,
                            length,
                            time_elapsed,
                        ]
                    )

               
                self.window_rewards.append(reward)
                self.window_lengths.append(length)

                if self.episode_count % self.window_size == 0:
                    avg_reward = sum(self.window_rewards) / len(self.window_rewards)
                    avg_length = sum(self.window_lengths) / len(self.window_lengths)

                    with open(self.avg_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                self.episode_count,
                                total_timesteps,
                                avg_reward,
                                avg_length,
                                time_elapsed,
                            ]
                        )

        
                    self.window_rewards.clear()
                    self.window_lengths.clear()

                    if self.verbose > 0:
                        print(
                            f"[EpisodeStats] Episodios {self.episode_count - self.window_size + 1}"
                            f"-{self.episode_count} | "
                            f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}"
                        )

        return True 


def main():
    ENV_ID = "ALE/MontezumaRevenge-v5"

    N_ENVS = 8

    TOTAL_TIMESTEPS = 50_000_000

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("best_model", exist_ok=True)

    train_env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=0)
    train_env = VecFrameStack(train_env, n_stack=4)

    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=123)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    model = PPO(
        "CnnPolicy",        
        train_env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=64,         
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,     
        device="cpu",        
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path="./checkpoints",
        name_prefix="ppo_montezuma",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=50_000 // N_ENVS,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    episode_stats_callback = EpisodeStatsCallback(
        log_dir="./logs",
        window_size=1000,
        verbose=1,
    )

    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, episode_stats_callback],
    )

    
    model.save("ppo_montezuma_final")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
