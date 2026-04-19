from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from tqdm import trange

from frozen_lake_env import FrozenLake8x8Stochastic


class StochasticModel:
    def __init__(self):
        self._counts: dict[tuple[int, int], dict[tuple[int, float, bool], int]] = \
            defaultdict(lambda: defaultdict(int))
        self._visited_sa: list[tuple[int, int]] = []
        self._visited_sa_set: set[tuple[int, int]] = set()

    def update(self, s: int, a: int, s_next: int, r: float, terminated: bool) -> None:
        key = (s, a)
        self._counts[key][(s_next, float(r), bool(terminated))] += 1
        if key not in self._visited_sa_set:
            self._visited_sa_set.add(key)
            self._visited_sa.append(key)

    def sample_sa(self, rng: np.random.Generator) -> tuple[int, int]:
        idx = int(rng.integers(0, len(self._visited_sa)))
        return self._visited_sa[idx]

    def sample_next(self, s: int, a: int,
                    rng: np.random.Generator) -> tuple[int, float, bool]:
        outcomes = self._counts[(s, a)]
        keys = list(outcomes.keys())
        weights = np.fromiter((outcomes[k] for k in keys), dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(keys), p=weights))
        return keys[idx]

    @property
    def n_visited(self) -> int:
        return len(self._visited_sa)


@dataclass
class DynaQConfig:
    num_episodes: int = 15_000
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9995
    planning_steps: int = 20
    slip_prob: float = 2.0 / 3.0
    q_init: float = 1.0
    seed: int = 42


def train_dyna_q(cfg: DynaQConfig):
    env = FrozenLake8x8Stochastic(slip_prob=cfg.slip_prob, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    Q = np.full((env.n_states, env.n_actions), cfg.q_init, dtype=np.float64)
    for s in env.terminal_states:
        Q[s] = 0.0
    model = StochasticModel()
    eps = cfg.eps_start

    rewards, lengths, successes = [], [], []

    for _ in trange(cfg.num_episodes, desc=f"Dyna-Q(n={cfg.planning_steps})"):
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

            model.update(state, action, next_state, reward, terminated)

            for _ in range(cfg.planning_steps):
                ps, pa = model.sample_sa(rng)
                pns, pr, p_term = model.sample_next(ps, pa, rng)
                if p_term:
                    ptd = pr
                else:
                    ptd = pr + cfg.gamma * float(np.max(Q[pns]))
                Q[ps, pa] += cfg.alpha * (ptd - Q[ps, pa])

            state = next_state
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        eps = max(cfg.eps_min, eps * cfg.eps_decay)
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(1 if total_reward > 0 else 0)

    return Q, rewards, lengths, successes, model


if __name__ == "__main__":
    cfg = DynaQConfig()
    Q, rewards, lengths, successes, model = train_dyna_q(cfg)
    avg_last = float(np.mean(rewards[-500:]))
    print(f"Visited (s,a) pairs: {model.n_visited}")
    print(f"Final success rate over last 500 episodes: {avg_last:.3f}")
