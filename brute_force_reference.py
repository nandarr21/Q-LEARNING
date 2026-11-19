import itertools
import numpy as np
from inventory_env import MinimarketInventoryEnv as Env, ACTION_SYMBOLS


def brute_force(n_episodes=300, seed=7):
    rng = np.random.default_rng(seed)
    T = 30
    policies = np.array(list(itertools.product(range(3), repeat=9)), dtype=np.int8)   # (P, 9)
    P = len(policies)
    # skenario acak bersama
    stock0 = rng.integers(0, Env.CAPACITY + 1, n_episodes)
    dcat = np.zeros((n_episodes, T + 1), dtype=int)
    dcat[:, 0] = rng.integers(0, 3, n_episodes)
    u = rng.random((n_episodes, T))
    cum = Env.DEMAND_TRANSITION.cumsum(axis=1)
    for t in range(T):
        dcat[:, t + 1] = (u[:, t][:, None] > cum[dcat[:, t]]).sum(axis=1)
    lo = np.array([v[0] for v in Env.DEMAND_RANGE.values()])
    hi = np.array([v[1] for v in Env.DEMAND_RANGE.values()])
    dem = (lo[dcat[:, :T]] + np.floor(rng.random((n_episodes, T)) * (hi - lo + 1)[dcat[:, :T]])).astype(int)

    def cat(x):
        return np.where(x <= 6, 0, np.where(x <= 13, 1, 2))

    order = np.array([Env.ORDER_QTY[a] for a in range(3)])
    stock = np.broadcast_to(stock0, (P, n_episodes)).copy()
    total = np.zeros((P, n_episodes))
    for t in range(T):
        state = cat(stock) * 3 + dcat[:, t][None, :]
        act = policies[np.arange(P)[:, None], state]
        s2 = stock + order[act]
        overflow = s2 > Env.CAPACITY
        s2 = np.minimum(s2, Env.CAPACITY)
        sold = np.minimum(s2, dem[:, t][None, :])
        full = sold == dem[:, t][None, :]
        stock = s2 - sold
        c = cat(stock)
        total += np.where(full, Env.R_FULFILLED, Env.R_STOCKOUT) + (c == 1) * Env.R_IDEAL \
                 + ((c == 2) | overflow) * Env.R_OVERSTOCK
    mean = total.mean(axis=1)
    order_idx = np.argsort(-mean)
    return policies, mean, order_idx


if __name__ == "__main__":
    pol, mean, idx = brute_force()
    print("10 kebijakan terbaik dari 19.683 kemungkinan:")
    for i in idx[:10]:
        print("".join(ACTION_SYMBOLS[a] for a in pol[i]), round(mean[i], 1))
    print("Kebijakan terburuk:", "".join(ACTION_SYMBOLS[a] for a in pol[idx[-1]]), round(mean[idx[-1]], 1))
