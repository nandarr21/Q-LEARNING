import numpy as np

STOCK_LABELS = ["Rendah", "Sedang", "Tinggi"]
DEMAND_LABELS = ["Rendah", "Sedang", "Tinggi"]
ACTION_LABELS = ["Pesan Sedikit", "Pesan Banyak", "Tidak Pesan"]
ACTION_SYMBOLS = ["S", "B", "-"]          # simbol singkat untuk peta kebijakan


class MinimarketInventoryEnv:
    """Simulasi persediaan satu jenis barang di minimarket selama satu periode (episode)."""

    # ---- parameter fisik minimarket (satuan: unit/karton) ----
    CAPACITY = 20                                 # kapasitas rak + gudang
    STOCK_BINS = [(0, 6), (7, 13), (14, 20)]      # batas kategori stok Rendah/Sedang/Tinggi
    ORDER_QTY = {0: 5, 1: 10, 2: 0}               # jumlah unit per aksi
    DEMAND_RANGE = {0: (2, 4), 1: (5, 8), 2: (9, 12)}   # jumlah permintaan per kategori
    # Peluang perubahan tingkat permintaan dari hari ke hari (baris = hari ini)
    DEMAND_TRANSITION = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ])

    # ---- reward sesuai TP-07 ----
    R_FULFILLED = 10
    R_IDEAL = 5
    R_STOCKOUT = -5
    R_OVERSTOCK = -3

    n_states = 9
    n_actions = 3

    def __init__(self, episode_length=30, seed=None):
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------- helper ----------
    @classmethod
    def stock_category(cls, units):
        for idx, (lo, hi) in enumerate(cls.STOCK_BINS):
            if lo <= units <= hi:
                return idx
        return 2

    @staticmethod
    def encode(stock_cat, demand_cat):
        """(stok, permintaan) -> indeks state 0..8  (state = stok*3 + permintaan)."""
        return stock_cat * 3 + demand_cat

    @staticmethod
    def decode(state):
        return divmod(state, 3)

    @staticmethod
    def state_label(state):
        s, d = divmod(state, 3)
        return f"Stok {STOCK_LABELS[s]} | Permintaan {DEMAND_LABELS[d]}"

    # ---------- API utama ----------
    def reset(self):
        self.day = 0
        self.stock = int(self.rng.integers(0, self.CAPACITY + 1))
        self.demand_cat = int(self.rng.integers(0, 3))
        self.state = self.encode(self.stock_category(self.stock), self.demand_cat)
        return self.state

    def step(self, action):
        """Menerima action -> mengembalikan (new_state, reward, done, info)."""
        assert action in (0, 1, 2), "Aksi harus 0, 1, atau 2"

        # 1) Barang pesanan datang di awal hari
        incoming = self.ORDER_QTY[action]
        stock_after_order = self.stock + incoming
        overflow = max(0, stock_after_order - self.CAPACITY)
        stock_after_order = min(stock_after_order, self.CAPACITY)

        # 2) Pelanggan datang: jumlah permintaan acak sesuai kategori hari ini
        lo, hi = self.DEMAND_RANGE[self.demand_cat]
        demand = int(self.rng.integers(lo, hi + 1))
        sold = min(stock_after_order, demand)
        self.stock = stock_after_order - sold
        end_cat = self.stock_category(self.stock)

        # 3) Hitung reward (rancangan TP-07)
        reward = 0
        fulfilled = sold == demand
        if fulfilled:
            reward += self.R_FULFILLED
        else:
            reward += self.R_STOCKOUT            # stok habis sebelum permintaan terpenuhi
        if end_cat == 1:
            reward += self.R_IDEAL               # persediaan ideal
        if end_cat == 2 or overflow > 0:
            reward += self.R_OVERSTOCK           # penumpukan barang

        # 4) Permintaan hari berikutnya & state baru
        self.demand_cat = int(self.rng.choice(3, p=self.DEMAND_TRANSITION[self.demand_cat]))
        self.state = self.encode(end_cat, self.demand_cat)
        self.day += 1
        done = self.day >= self.episode_length

        info = {"demand": demand, "sold": sold, "lost_sales": demand - sold,
                "order": incoming, "overflow": overflow, "stock_units": self.stock,
                "fulfilled": fulfilled}
        return self.state, reward, done, info
