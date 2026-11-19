# TA-07 — Sistem Pengelolaan Persediaan Barang pada Minimarket Menggunakan Q-Learning

Tugas Akhir Praktikum Pembelajaran Mesin, Modul 07 (Reinforcement Learning). Proyek ini melanjutkan rancangan TP-07.

## Ringkasan Skenario
| Elemen | Definisi |
|---|---|
| State (9) | Stok {Rendah, Sedang, Tinggi} × Permintaan {Rendah, Sedang, Tinggi} |
| Action (3) | Pesan Sedikit (5 unit), Pesan Banyak (10 unit), Tidak Pesan |
| Reward | +10 permintaan terpenuhi, +5 stok ideal, −5 stok habis, −3 penumpukan |
| Episode | 30 hari, kapasitas gudang 20 unit |

## Struktur Repositori
```
├── TA07_QLearning_Persediaan_Minimarket.ipynb   # Langkah 1–6 lengkap dengan output & analisis
├── inventory_env.py            # Lingkungan simulasi (state, action -> new_state, reward, done)
├── q_agent.py                  # Q-Learning + epsilon-greedy dan fungsi evaluasi kebijakan
├── eksperimen_hiperparameter.py# Langkah 5: eksperimen learning rate (alpha)
├── brute_force_reference.py    # Validasi: evaluasi seluruh 3^9 kebijakan sebagai acuan optimum
├── figures/                    # Grafik hasil (kurva, kebijakan, eksperimen, simulasi)
├── results/                    # Q-Table final, tabel evaluasi, hasil eksperimen (CSV)
└── requirements.txt
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
jupyter notebook TA07_QLearning_Persediaan_Minimarket.ipynb   # atau unggah ke Google Colab
python eksperimen_hiperparameter.py                           # eksperimen alpha (Langkah 5)
python brute_force_reference.py                               # kebijakan acuan (opsional)
```
Notebook bersifat mandiri (semua kelas didefinisikan di dalamnya), sehingga bisa langsung dijalankan di Google Colab tanpa mengunggah file `.py`.

## Hasil Utama
| Model | α | γ | decay ε | Episode | Reward evaluasi | Peringkat dari 19.683 kebijakan |
|---|---|---|---|---|---|---|
| Baseline | 0.8 | 0.9 | 0.005 | 2000 | 352,7 | 277 |
| **Final** | **0.1** | 0.9 | 0.002 | 3000 | **389,3** | **21** (≈98,5% dari optimum) |

Kebijakan final: stok rendah → pesan banyak; stok sedang → jumlah pesanan disesuaikan dengan permintaan; stok tinggi dan permintaan rendah → tidak pesan.
Eksperimen α menunjukkan bahwa α = 0.8 membuat Q-Table tidak stabil pada permintaan acak, sedangkan α = 0.01 membuat nilai Q tumbuh lambat.
