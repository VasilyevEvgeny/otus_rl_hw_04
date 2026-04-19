from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import trange

from frozen_lake_env import FrozenLake8x8Stochastic


@dataclass
class QLearningConfig:
    num_episodes: int = 15_000
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9995
    slip_prob: float = 2.0 / 3.0
    q_init: float = 1.0
    seed: int = 42


def train_q_learning(cfg: QLearningConfig):
    env = FrozenLake8x8Stochastic(slip_prob=cfg.slip_prob, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    Q = np.full((env.n_states, env.n_actions), cfg.q_init, dtype=np.float64)
    for s in env.terminal_states:
        Q[s] = 0.0
    eps = cfg.eps_start

    rewards, lengths, successes = [], [], []

    for _ in trange(cfg.num_episodes, desc="Q-learning"):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(cfg.max_steps):
            if rng.random() < eps:
                action = int(rng.integers(0, env.n_actions))
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                td_target = reward
            else:
                td_target = reward + cfg.gamma * float(np.max(Q[next_state]))
            Q[state, action] += cfg.alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        eps = max(cfg.eps_min, eps * cfg.eps_decay)
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(1 if total_reward > 0 else 0)

    return Q, rewards, lengths, successes


if __name__ == "__main__":
    cfg = QLearningConfig()
    Q, rewards, lengths, successes = train_q_learning(cfg)
    avg_last = float(np.mean(rewards[-500:]))
    print(f"Final success rate over last 500 episodes: {avg_last:.3f}")
