from __future__ import annotations

import numpy as np


MAP_8x8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = ["Left", "Down", "Right", "Up"]
ACTION_DELTAS = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}


class FrozenLake8x8Stochastic:
    def __init__(self, desc: list[str] | None = None, slip_prob: float = 2.0 / 3.0,
                 seed: int | None = None):
        desc = desc if desc is not None else MAP_8x8
        self.desc = np.array([list(row) for row in desc])
        self.n_rows, self.n_cols = self.desc.shape
        self.n_states = int(self.n_rows * self.n_cols)
        self.n_actions = 4
        self.slip_prob = float(slip_prob)

        self.rng = np.random.default_rng(seed)

        self.start_rc = self._find("S")
        self.goal_rc = self._find("G")

        self.terminal_states = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.desc[r, c] in ("H", "G"):
                    self.terminal_states.add(self._rc_to_state(r, c))

        self.state = self._rc_to_state(*self.start_rc)

    def _find(self, ch: str) -> tuple[int, int]:
        rs, cs = np.where(self.desc == ch)
        return int(rs[0]), int(cs[0])

    def _rc_to_state(self, r: int, c: int) -> int:
        return int(r * self.n_cols + c)

    def _state_to_rc(self, s: int) -> tuple[int, int]:
        return int(s // self.n_cols), int(s % self.n_cols)

    def _step_deterministic(self, r: int, c: int, action: int) -> tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        nr = min(max(r + dr, 0), self.n_rows - 1)
        nc = min(max(c + dc, 0), self.n_cols - 1)
        return nr, nc

    def _transition_probs(self, action: int) -> np.ndarray:
        intended = 1.0 - self.slip_prob
        perp_a = (action - 1) % 4
        perp_b = (action + 1) % 4
        probs = np.zeros(4, dtype=np.float64)
        probs[action] += intended
        probs[perp_a] += self.slip_prob / 2.0
        probs[perp_b] += self.slip_prob / 2.0
        return probs

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self._rc_to_state(*self.start_rc)
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if self.state in self.terminal_states:
            return self.state, 0.0, True, False, {}

        probs = self._transition_probs(int(action))
        actual = int(self.rng.choice(4, p=probs))

        r, c = self._state_to_rc(self.state)
        nr, nc = self._step_deterministic(r, c, actual)
        next_state = self._rc_to_state(nr, nc)
        cell = self.desc[nr, nc]

        if cell == "G":
            reward, terminated = 1.0, True
        elif cell == "H":
            reward, terminated = 0.0, True
        else:
            reward, terminated = 0.0, False

        self.state = next_state
        return next_state, reward, terminated, False, {}

    def true_transition_matrix(self) -> dict[tuple[int, int], list[tuple[float, int, float, bool]]]:
        matrix: dict[tuple[int, int], list[tuple[float, int, float, bool]]] = {}
        for s in range(self.n_states):
            r, c = self._state_to_rc(s)
            if s in self.terminal_states:
                for a in range(self.n_actions):
                    matrix[(s, a)] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.n_actions):
                probs = self._transition_probs(a)
                outcomes: dict[tuple[int, float, bool], float] = {}
                for actual in range(4):
                    p = probs[actual]
                    if p == 0.0:
                        continue
                    nr, nc = self._step_deterministic(r, c, actual)
                    ns = self._rc_to_state(nr, nc)
                    cell = self.desc[nr, nc]
                    if cell == "G":
                        rew, term = 1.0, True
                    elif cell == "H":
                        rew, term = 0.0, True
                    else:
                        rew, term = 0.0, False
                    key = (ns, rew, term)
                    outcomes[key] = outcomes.get(key, 0.0) + p
                matrix[(s, a)] = [(p, ns, rew, term) for (ns, rew, term), p in outcomes.items()]
        return matrix


if __name__ == "__main__":
    env = FrozenLake8x8Stochastic(seed=0)
    print(f"States: {env.n_states}, actions: {env.n_actions}")
    print(f"Start: {env.start_rc}, goal: {env.goal_rc}")
    print(f"Terminal states: {sorted(env.terminal_states)}")

    state, _ = env.reset(seed=0)
    for t in range(10):
        action = env.rng.integers(0, env.n_actions)
        state, reward, terminated, truncated, _ = env.step(int(action))
        print(f"step={t} action={ACTION_NAMES[action]} "
              f"state={state} reward={reward} terminated={terminated}")
        if terminated or truncated:
            break
