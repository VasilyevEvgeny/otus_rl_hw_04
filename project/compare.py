from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dyna_q import DynaQConfig, train_dyna_q
from q_learning import QLearningConfig, train_q_learning


RESULTS_DIR = "results"
ROLLING_WINDOW = 200


@dataclass
class RunConfig:
    num_episodes: int = 15_000
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9995
    slip_prob: float = 2.0 / 3.0
    q_init: float = 1.0
    planning_steps: tuple[int, ...] = (5, 20, 50)
    seed: int = 42


def rolling_mean(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = values[lo:i + 1].mean()
    return out


def evaluate_greedy(Q: np.ndarray, slip_prob: float, seed: int,
                    n_episodes: int = 500, max_steps: int = 200) -> float:
    from frozen_lake_env import FrozenLake8x8Stochastic

    env = FrozenLake8x8Stochastic(slip_prob=slip_prob, seed=seed)
    total = 0.0
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
    return total / n_episodes


def value_iteration(env, gamma: float, tol: float = 1e-9,
                    max_iter: int = 5_000) -> np.ndarray:
    P = env.true_transition_matrix()
    V = np.zeros(env.n_states, dtype=np.float64)
    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            best = -np.inf
            for a in range(env.n_actions):
                q = 0.0
                for p, ns, r, term in P[(s, a)]:
                    q += p * (r + gamma * (0.0 if term else V[ns]))
                if q > best:
                    best = q
            V_new[s] = best
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    Q_star = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            Q_star[s, a] = sum(p * (r + gamma * (0.0 if term else V[ns]))
                               for p, ns, r, term in P[(s, a)])
    return Q_star


def plot_comparison(results: dict[str, dict], cfg: RunConfig,
                    optimal_success: float | None = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    episodes = np.arange(1, cfg.num_episodes + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        ax.plot(episodes, rolling_mean(data["successes"], ROLLING_WINDOW),
                linewidth=1.5, label=name)
    if optimal_success is not None:
        ax.axhline(optimal_success, color="black", linestyle="--", linewidth=1.2,
                   label=f"Optimal policy ceiling = {optimal_success:.3f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success rate (rolling avg, window={ROLLING_WINDOW})")
    ax.set_title("FrozenLake 8x8 (slippery): Q-learning vs Dyna-Q")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "success_rate.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        ax.plot(episodes, rolling_mean(data["lengths"], ROLLING_WINDOW),
                linewidth=1.5, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode length (rolling avg, window={ROLLING_WINDOW})")
    ax.set_title("FrozenLake 8x8 (slippery): episode length comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "episode_length.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in results.items():
        cum = np.cumsum(data["rewards"])
        ax.plot(episodes, cum, linewidth=1.5, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward (goal reaches)")
    ax.set_title("FrozenLake 8x8 (slippery): cumulative goal reaches")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cumulative_reward.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {RESULTS_DIR}/")


def plot_policy(Q: np.ndarray, env, name: str, filepath: str):
    from frozen_lake_env import ACTION_DELTAS

    n_rows, n_cols = env.n_rows, env.n_cols
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(n_rows):
        for c in range(n_cols):
            cell = env.desc[r, c]
            color = {
                "S": "lightgreen",
                "G": "gold",
                "H": "dimgray",
                "F": "lightblue",
            }[cell]
            ax.add_patch(plt.Rectangle((c, n_rows - r - 1), 1, 1,
                                       facecolor=color, edgecolor="black"))
            ax.text(c + 0.15, n_rows - r - 0.85, cell, fontsize=8, color="black")

    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            if env.desc[r, c] in ("H", "G"):
                continue
            a = int(np.argmax(Q[s]))
            dr, dc = ACTION_DELTAS[a]
            x = c + 0.5
            y = n_rows - r - 0.5
            ax.arrow(x, y, dc * 0.3, -dr * 0.3,
                     head_width=0.15, head_length=0.12,
                     fc="black", ec="black", length_includes_head=True)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Greedy policy: {name}")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = RunConfig()

    shared = dict(
        num_episodes=cfg.num_episodes,
        max_steps=cfg.max_steps,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        eps_start=cfg.eps_start,
        eps_min=cfg.eps_min,
        eps_decay=cfg.eps_decay,
        slip_prob=cfg.slip_prob,
        q_init=cfg.q_init,
        seed=cfg.seed,
    )

    results: dict[str, dict] = {}

    print("=== Training Q-learning baseline ===")
    ql_cfg = QLearningConfig(**shared)
    Q_ql, r_ql, l_ql, s_ql = train_q_learning(ql_cfg)
    results["Q-learning"] = {
        "rewards": r_ql, "lengths": l_ql, "successes": s_ql, "Q": Q_ql,
    }

    for n in cfg.planning_steps:
        print(f"=== Training Dyna-Q (n={n}) ===")
        dq_cfg = DynaQConfig(planning_steps=n, **shared)
        Q_dq, r_dq, l_dq, s_dq, _ = train_dyna_q(dq_cfg)
        results[f"Dyna-Q (n={n})"] = {
            "rewards": r_dq, "lengths": l_dq, "successes": s_dq, "Q": Q_dq,
        }

    print("\n=== Greedy evaluation (500 episodes each) ===")
    summary_lines = []
    for name, data in results.items():
        succ = evaluate_greedy(data["Q"], cfg.slip_prob, seed=cfg.seed + 10_000)
        final_train = float(np.mean(data["successes"][-500:]))
        summary_lines.append(
            f"{name:>18s} | train last-500 success: {final_train:.3f} | "
            f"greedy eval success: {succ:.3f}"
        )
        print(summary_lines[-1])

    from frozen_lake_env import FrozenLake8x8Stochastic
    env = FrozenLake8x8Stochastic(slip_prob=cfg.slip_prob, seed=cfg.seed)

    print("\n=== Value iteration on true transition matrix ===")
    Q_star = value_iteration(env, cfg.gamma)
    np.save(os.path.join(RESULTS_DIR, "Q_optimal.npy"), Q_star)
    optimal_success = evaluate_greedy(Q_star, cfg.slip_prob,
                                      seed=cfg.seed + 10_000,
                                      n_episodes=2_000)
    summary_lines.append(
        f"{'Optimal (VI)':>18s} | greedy eval success (2000 ep): {optimal_success:.3f}"
    )
    print(summary_lines[-1])
    plot_policy(Q_star, env, "Optimal (VI)",
                os.path.join(RESULTS_DIR, "policy_Optimal_VI.png"))

    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    plot_comparison(results, cfg, optimal_success=optimal_success)
    for name, data in results.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_policy(data["Q"], env, name,
                    os.path.join(RESULTS_DIR, f"policy_{safe}.png"))
        np.save(os.path.join(RESULTS_DIR, f"Q_{safe}.npy"), data["Q"])

    np.savez(os.path.join(RESULTS_DIR, "training_curves.npz"),
             **{f"{name}__{metric}": np.asarray(data[metric])
                for name, data in results.items()
                for metric in ("rewards", "lengths", "successes")})

    print(f"\nAll artifacts saved under {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
