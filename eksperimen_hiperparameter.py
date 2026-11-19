import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inventory_env import MinimarketInventoryEnv, ACTION_SYMBOLS
from q_agent import train_q_learning, evaluate_policy

ALPHAS = {"α = 0.8 (default, tinggi)": 0.8,
          "α = 0.01 (ekstrem rendah)": 0.01,
          "α = 0.1 (pembanding)": 0.1}
COLORS = {0.8: "#d62728", 0.01: "#1f77b4", 0.1: "#2ca02c"}
SEEDS = [0, 1, 2, 3, 4]
GAMMA, EPISODES, WINDOW = 0.9, 2000, 100


def policy_string(q_table):
    """Kebijakan 9 state sebagai string, urutan state 0..8 (S=sedikit, B=banyak, -=tidak pesan)."""
    return "".join(ACTION_SYMBOLS[a] for a in q_table.argmax(axis=1))


def run_experiment(alphas=ALPHAS, seeds=SEEDS, episodes=EPISODES, gamma=GAMMA):
    results = {}
    for label, alpha in alphas.items():
        runs = []
        for seed in seeds:
            q, hist = train_q_learning(alpha=alpha, gamma=gamma, episodes=episodes, seed=seed)
            ev = evaluate_policy(lambda s, q=q: int(np.argmax(q[s])), n_episodes=300)
            runs.append({"seed": seed, "q": q, "hist": hist, "eval": ev, "policy": policy_string(q)})
        results[label] = {"alpha": alpha, "runs": runs}
    return results


def summarize(results, window=WINDOW):
    rows, pol_rows = [], []
    for label, res in results.items():
        rewards = np.array([r["hist"]["rewards"] for r in res["runs"]])
        ma = pd.Series(rewards.mean(axis=0)).rolling(window, min_periods=1).mean()
        last = rewards[:, -500:]
        policies = [r["policy"] for r in res["runs"]]
        rows.append({
            "Konfigurasi": label,
            "MA reward ep-200": ma.iloc[199],
            "MA reward ep-500": ma.iloc[499],
            "MA reward akhir": ma.iloc[-1],
            "Std reward 500 ep terakhir": last.std(axis=1).mean(),
            "Reward evaluasi greedy (rata2)": np.mean([r["eval"]["Rata-rata reward/episode"] for r in res["runs"]]),
            "Reward evaluasi greedy (std antar seed)": np.std([r["eval"]["Rata-rata reward/episode"] for r in res["runs"]]),
            "Rata2 max Q akhir": np.mean([r["hist"]["q_mean"][-1] for r in res["runs"]]),
            "Jumlah kebijakan unik (dari 5 seed)": len(set(policies)),
        })
        for r in res["runs"]:
            pol_rows.append({"Konfigurasi": label, "seed": r["seed"], "Kebijakan (state 0-8)": r["policy"],
                             "Reward evaluasi": r["eval"]["Rata-rata reward/episode"]})
    return pd.DataFrame(rows).round(2), pd.DataFrame(pol_rows).round(1)


def plot_experiment(results, window=WINDOW, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label, res in results.items():
        a, c = res["alpha"], COLORS.get(res["alpha"], None)
        rewards = np.array([r["hist"]["rewards"] for r in res["runs"]])
        qmean = np.array([r["hist"]["q_mean"] for r in res["runs"]])
        qchg = np.array([r["hist"]["q_change"] for r in res["runs"]])
        style = "--" if a == 0.1 else "-"
        axes[0].plot(pd.Series(rewards.mean(0)).rolling(window, min_periods=1).mean(), style, color=c, label=label)
        axes[1].plot(qmean.mean(0), style, color=c, label=label)
        axes[2].plot(pd.Series(qchg.mean(0)).rolling(50, min_periods=1).mean(), style, color=c, label=label)

    axes[0].set_title(f"Kurva Pembelajaran (MA {window}, rata-rata 5 seed)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Rata-rata total reward / episode")
    axes[1].set_title("Pertumbuhan Nilai Q (rata-rata max Q tiap state)")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Nilai Q")
    axes[2].set_title("Stabilitas Q-Table (total |ΔQ| per episode, MA 50)")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("|ΔQ|"); axes[2].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right"); axes[1].legend(fontsize=9, loc="lower right"); axes[2].legend(fontsize=9, loc="center right")
    fig.suptitle("Eksperimen Hiperparameter: Pengaruh Learning Rate (α), γ = 0.9", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    res = run_experiment()
    summary, policies = summarize(res)
    summary.to_csv("results/eksperimen_alpha_ringkasan.csv", index=False)
    policies.to_csv("results/eksperimen_alpha_kebijakan.csv", index=False)
    plot_experiment(res, save_path="figures/eksperimen_alpha.png")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print()
    print(policies.to_string(index=False))
    print("\nGrafik disimpan di figures/eksperimen_alpha.png")
