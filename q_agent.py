import numpy as np
from inventory_env import MinimarketInventoryEnv


def train_q_learning(alpha=0.8, gamma=0.9, episodes=2000, epsilon=1.0, min_epsilon=0.01,
                     max_epsilon=1.0, decay_rate=0.005, episode_length=30, seed=42):
    """
    Melatih agen Q-Learning pada MinimarketInventoryEnv.

    Return
    ------
    q_table : ndarray (9, 3)
    history : dict berisi
        rewards  - total reward tiap episode
        epsilons - nilai epsilon yang dipakai tiap episode
        q_mean   - rata-rata max_a Q(s,a) di akhir tiap episode (memantau pertumbuhan nilai Q)
        q_change - total |perubahan Q| dalam satu episode (memantau stabilitas Q-Table)
        visits   - jumlah kunjungan tiap state selama pelatihan
    """
    env = MinimarketInventoryEnv(episode_length=episode_length, seed=seed)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions))
    history = {"rewards": [], "epsilons": [], "q_mean": [], "q_change": [],
               "visits": np.zeros(env.n_states, dtype=int)}

    for episode in range(episodes):
        state = env.reset()
        total_reward, total_change, done = 0, 0.0, False

        while not done:
            # 1. Pilih aksi dengan epsilon-greedy
            if rng.uniform() < epsilon:
                action = int(rng.integers(env.n_actions))          # eksplorasi
            else:
                action = int(np.argmax(q_table[state]))           # eksploitasi

            # 2. Jalankan aksi di lingkungan
            new_state, reward, done, _ = env.step(action)

            # 3. Update Q-Table (Persamaan Bellman)
            #    Q(s,a) <- (1-alpha)Q(s,a) + alpha[r + gamma * max_a' Q(s',a')]
            #    Pada hari terakhir (terminal) tidak ada nilai masa depan.
            target = reward if done else reward + gamma * np.max(q_table[new_state])
            old_q = q_table[state, action]
            q_table[state, action] = (1 - alpha) * old_q + alpha * target

            total_change += abs(q_table[state, action] - old_q)
            history["visits"][state] += 1
            total_reward += reward
            state = new_state

        history["rewards"].append(total_reward)
        history["epsilons"].append(epsilon)
        history["q_mean"].append(q_table.max(axis=1).mean())
        history["q_change"].append(total_change)

        # 4. Peluruhan epsilon (eksplorasi berkurang seiring waktu)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    return q_table, history


def evaluate_policy(policy_fn, n_episodes=500, episode_length=30, seed=2024):
    """Menjalankan kebijakan tanpa eksplorasi (greedy) dan merangkum performanya."""
    env = MinimarketInventoryEnv(episode_length=episode_length, seed=seed)
    totals, days = [], 0
    counts = {"fulfilled": 0, "stockout": 0, "ideal": 0, "overstock": 0, "lost": 0, "demand": 0}
    for _ in range(n_episodes):
        state, done, total = env.reset(), False, 0
        while not done:
            state, r, done, info = env.step(policy_fn(state))
            total += r
            days += 1
            cat = env.stock_category(info["stock_units"])
            counts["fulfilled"] += info["fulfilled"]
            counts["stockout"] += not info["fulfilled"]
            counts["ideal"] += cat == 1
            counts["overstock"] += (cat == 2) or info["overflow"] > 0
            counts["lost"] += info["lost_sales"]
            counts["demand"] += info["demand"]
        totals.append(total)
    return {
        "Rata-rata reward/episode": np.mean(totals),
        "Service level (%)": 100 * counts["fulfilled"] / days,
        "Hari stok habis (%)": 100 * counts["stockout"] / days,
        "Hari stok ideal (%)": 100 * counts["ideal"] / days,
        "Hari penumpukan (%)": 100 * counts["overstock"] / days,
        "Penjualan hilang (%)": 100 * counts["lost"] / counts["demand"],
    }
