import csv
import time
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

import torch
from torch import optim

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteSACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

ALL_REWARDS = []

REWARD_LOG_FILE = None
REWARD_CSV_WRITER = None
REWARD_STEP_INDEX = 0


class RewardRecorder(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        global ALL_REWARDS, REWARD_CSV_WRITER, REWARD_LOG_FILE, REWARD_STEP_INDEX
        r = float(reward)
        ALL_REWARDS.append(r)

        if REWARD_CSV_WRITER is not None and REWARD_LOG_FILE is not None:
            REWARD_CSV_WRITER.writerow([REWARD_STEP_INDEX, r])
            REWARD_LOG_FILE.flush()

        REWARD_STEP_INDEX = REWARD_STEP_INDEX + 1
        return reward


def make_env(render_mode=None, seed=None):
    env = gym.make("MontezumaRevengeNoFrameskip-v4", render_mode=render_mode)

    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )

    env = FrameStack(env, 4)
    env = RewardRecorder(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def main():
    device = "cpu"

    buffer_size = 500000
    TOTAL_STEPS = 5_000_000
    STEP_LOG_INTERVAL = 1000
    EPISODE_LOG_INTERVAL = 1000
    BATCH_SIZE = 64
    UPDATE_PER_STEP = 1.0
    COLLECT_N_STEP = 10

    global REWARD_LOG_FILE, REWARD_CSV_WRITER, REWARD_STEP_INDEX
    REWARD_STEP_INDEX = 0
    REWARD_LOG_FILE = open("__all_rewards_stream.csv", "w", newline="")
    REWARD_CSV_WRITER = csv.writer(REWARD_LOG_FILE)
    REWARD_CSV_WRITER.writerow(["step_index", "reward"])
    REWARD_LOG_FILE.flush()

    score_log_file = open("_score_vs_steps_stream.csv", "w", newline="")
    score_writer = csv.writer(score_log_file)
    score_writer.writerow(["global_step", "avg_return_last_50_eps", "elapsed_time_sec"])
    score_log_file.flush()

    len_log_file = open("_avg_length_vs_episodes_stream.csv", "w", newline="")
    len_writer = csv.writer(len_log_file)
    len_writer.writerow(["global_episode", "avg_len_last_1000_eps", "elapsed_time_sec"])
    len_log_file.flush()

    try:
        train_envs = DummyVectorEnv([lambda: make_env()])

        tmp_env = make_env()
        state_shape = tmp_env.observation_space.shape
        action_space = tmp_env.action_space
        action_shape = action_space.n
        tmp_env.close()

        actor_base = Net(
            state_shape=state_shape,
            hidden_sizes=[256, 256],
            device=device,
        )

        actor = Actor(
            preprocess_net=actor_base,
            action_shape=[action_shape],
            hidden_sizes=[],
            softmax_output=False,
            device=device,
        ).to(device)

        actor_optim = optim.Adam(actor.parameters(), lr=3e-4)

        critic_base1 = Net(
            state_shape=state_shape,
            hidden_sizes=[256, 256],
            device=device,
        )
        critic1 = Critic(
            preprocess_net=critic_base1,
            hidden_sizes=[],
            last_size=action_shape,
            device=device,
        ).to(device)
        critic1_optim = optim.Adam(critic1.parameters(), lr=3e-4)

        critic_base2 = Net(
            state_shape=state_shape,
            hidden_sizes=[256, 256],
            device=device,
        )
        critic2 = Critic(
            preprocess_net=critic_base2,
            hidden_sizes=[],
            last_size=action_shape,
            device=device,
        ).to(device)
        critic2_optim = optim.Adam(critic2.parameters(), lr=3e-4)

        policy = DiscreteSACPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic1=critic1,
            critic1_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=0.005,
            gamma=0.99,
            alpha=0.2,
            reward_normalization=False,
            estimation_step=1,
            action_space=action_space,
        ).to(device)

        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(buffer_size, len(train_envs)),
        )

        global_step = 0
        global_episode = 0
        next_step_log = STEP_LOG_INTERVAL

        all_ep_returns = []
        all_ep_lens = []

        start_time = time.time()

        while global_step < TOTAL_STEPS:
            result = train_collector.collect(n_step=COLLECT_N_STEP)

            n_st = result["n/st"]
            n_ep = result["n/ep"]

            if n_st == 0:
                break

            global_step = global_step + n_st

            if n_ep > 0:
                rews = result.get("rews", [])
                lens = result.get("lens", [])
                rews_flat = list(np.array(rews).flatten())
                lens_flat = list(np.array(lens).flatten())
                for v in rews_flat:
                    all_ep_returns.append(v)
                for v in lens_flat:
                    all_ep_lens.append(v)
                global_episode = global_episode + n_ep

            if len(train_collector.buffer) >= BATCH_SIZE:
                num_updates = int(UPDATE_PER_STEP * n_st)
                for i in range(num_updates):
                    policy.update(BATCH_SIZE, train_collector.buffer)

            if global_step >= next_step_log:
                elapsed = time.time() - start_time
                if len(all_ep_returns) > 0:
                    last_returns = all_ep_returns[-50:]
                    avg_return = float(np.mean(last_returns))
                else:
                    avg_return = 0.0

                score_writer.writerow([global_step, avg_return, elapsed])
                score_log_file.flush()

                next_step_log = next_step_log + STEP_LOG_INTERVAL

            if global_episode > 0 and (global_episode % EPISODE_LOG_INTERVAL == 0):
                if len(all_ep_lens) >= EPISODE_LOG_INTERVAL:
                    last_lens = all_ep_lens[-EPISODE_LOG_INTERVAL:]
                    avg_len_1000 = float(np.mean(last_lens))
                else:
                    avg_len_1000 = float(np.mean(all_ep_lens))

                elapsed2 = time.time() - start_time
                len_writer.writerow([global_episode, avg_len_1000, elapsed2])
                len_log_file.flush()

    finally:
        if REWARD_LOG_FILE is not None:
            REWARD_LOG_FILE.close()
        score_log_file.close()
        len_log_file.close()


if __name__ == "__main__":
    main()
