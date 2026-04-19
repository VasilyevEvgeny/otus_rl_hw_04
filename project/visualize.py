from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from frozen_lake_env import ACTION_NAMES, FrozenLake8x8Stochastic


OUTPUT_DIR = "results/gifs"
logger = logging.getLogger(__name__)


CELL_COLORS = {
    "S": "lightgreen",
    "G": "gold",
    "H": "dimgray",
    "F": "lightblue",
}


def render_frame(ax, env, trail, step, action, reward, total_reward, result):
    ax.clear()
    n_rows, n_cols = env.n_rows, env.n_cols

    for r in range(n_rows):
        for c in range(n_cols):
            cell = env.desc[r, c]
            ax.add_patch(plt.Rectangle((c, n_rows - r - 1), 1, 1,
                                       facecolor=CELL_COLORS[cell], edgecolor="black"))

    for (tr, tc) in trail[:-1]:
        ax.add_patch(plt.Circle((tc + 0.5, n_rows - tr - 0.5), 0.12,
                                facecolor="royalblue", alpha=0.45))

    if trail:
        r, c = trail[-1]
        ax.add_patch(plt.Circle((c + 0.5, n_rows - r - 0.5), 0.3,
                                facecolor="red", edgecolor="black"))
        ax.text(c + 0.5, n_rows - r - 0.5, "A", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    action_str = ACTION_NAMES[action] if action is not None else "Start"
    title = (f"Step {step} | Action: {action_str} | "
             f"Reward: {reward:+.0f} | Total: {total_reward:+.0f} | {result}")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def roll_greedy_episode(Q: np.ndarray, env: FrozenLake8x8Stochastic,
                        seed: int, max_steps: int = 200):
    state, _ = env.reset(seed=seed)
    r, c = env._state_to_rc(state)
    frames = [{
        "trail": [(r, c)],
        "step": 0, "action": None, "reward": 0.0, "total_reward": 0.0,
        "result": "running",
    }]

    total = 0.0
    for t in range(1, max_steps + 1):
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        r, c = env._state_to_rc(state)
        prev_trail = list(frames[-1]["trail"])
        prev_trail.append((r, c))

        if terminated and reward > 0:
            result = "GOAL"
        elif terminated:
            result = "HOLE"
        elif truncated:
            result = "TRUNC"
        else:
            result = "running"

        frames.append({
            "trail": prev_trail,
            "step": t, "action": action, "reward": reward,
            "total_reward": total, "result": result,
        })

        if terminated or truncated:
            break

    return frames, total


def save_gif(frames, env, filepath, fps=2, hold_start=1.5, hold_end=2.5):
    fig, ax = plt.subplots(figsize=(6, 6.3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08)

    legend_items = [
        mpatches.Patch(facecolor="lightgreen", edgecolor="black", label="Start"),
        mpatches.Patch(facecolor="gold", edgecolor="black", label="Goal"),
        mpatches.Patch(facecolor="lightblue", edgecolor="black", label="Frozen"),
        mpatches.Patch(facecolor="dimgray", edgecolor="black", label="Hole"),
        mpatches.Patch(facecolor="red", edgecolor="black", label="Agent"),
    ]
    fig.legend(handles=legend_items, loc="lower center", fontsize=7,
               framealpha=0.85, handlelength=1.5, ncol=5,
               bbox_to_anchor=(0.5, 0.01))

    start_pad = max(1, int(round(hold_start * fps)))
    end_pad = max(1, int(round(hold_end * fps)))
    padded = [frames[0]] * start_pad + list(frames) + [frames[-1]] * end_pad

    def update(i):
        f = padded[i]
        render_frame(ax, env, f["trail"], f["step"], f["action"],
                     f["reward"], f["total_reward"], f["result"])

    anim = FuncAnimation(fig, update, frames=len(padded), interval=500, repeat=False)
    anim.save(filepath, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--q-table", type=str, default="results/Q_Dyna-Q_n20.npy",
                        help="Path to saved Q-table")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Explicit list of seeds, one per episode; overrides --seed")
    parser.add_argument("--slip-prob", type=float, default=2.0 / 3.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--name", type=str, default=None,
                        help="Name tag used in GIF filenames")
    args = parser.parse_args()

    if not os.path.exists(args.q_table):
        logger.error("Q-table not found at %s", args.q_table)
        logger.error("Run compare.py first to train agents and save Q-tables.")
        sys.exit(1)

    Q = np.load(args.q_table)
    name = args.name or os.path.splitext(os.path.basename(args.q_table))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = FrozenLake8x8Stochastic(slip_prob=args.slip_prob, seed=args.seed)

    if args.seeds:
        episode_seeds = list(args.seeds)
    else:
        rng = np.random.default_rng(args.seed)
        episode_seeds = [int(rng.integers(0, 2 ** 31)) for _ in range(args.episodes)]

    for i, seed in enumerate(episode_seeds):
        frames, total = roll_greedy_episode(Q, env, seed=seed)
        result = frames[-1]["result"]
        steps = frames[-1]["step"]
        logger.info("Episode %d (seed=%d): steps=%d reward=%+.0f result=%s",
                    i + 1, seed, steps, total, result)
        gif_path = os.path.join(OUTPUT_DIR, f"{name}_episode_{i + 1}.gif")
        save_gif(frames, env, gif_path, fps=args.fps)
        logger.info("  GIF saved: %s", gif_path)


if __name__ == "__main__":
    main()
