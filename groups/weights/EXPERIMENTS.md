# Training Experiments Log

Tracking CNN training configurations, loss values, and gameplay results.

## Setup

- **Architecture**: 2-channel input → Conv2d(16, 3×3) → Conv2d(32, 3×3) → Linear(64) → Linear(7) with tanh output
- **Loss**: Weighted MSE — `log1p(visits)` per record, so high-visit positions dominate gradient
- **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4 (unless noted)
- **Evaluation**: 10 games vs each opponent, alternating Red/Yellow
- **Search depth**: 4 (negamax + alpha-beta in Mojo)

---

## Datasets

| ID | File | Total records | Notes |
|---|---|---|---|
| DS1 | `cnn-training/q_values` | ~23M | Original dataset, used for v1 |
| DS2 | `training/q_values` | ~40M → 55M+ | Growing dataset from active MCTS runner |

---

## Results Table

| Run | Dataset | min_visits | Epochs | Samples used | Val loss | vs Minimax d2 | vs Minimax d3 | vs v1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **v1** (baseline) | DS1 (23M) | 2 | 10 | 590K | 0.2508 | 100% as Red | 100% as Red | baseline | User-trained, best model so far |
| v2-a | DS2 (40M) | 2 | 10 | 1.07M | 0.2472 | 100% as Red | 0% | 0% | Same settings as v1 but worse data quality |
| v2-b | DS2 (40M) | 5 | 20 | 293K | 0.1035 | — | 0% | 0% | Lower loss, but narrow coverage → loses to v1 |
| v2-c | DS2 (40M) | 1 | 10 | 7.6M | 0.6722 | — | — | — | Too noisy — single-visit records dominate |
| v2-d | DS2 (40M) | 3 | 20 | ~500K | 0.1664 | — | 0% | 50/50 | Equal to v1 direct, loses to minimax as Red |
| v2-e | DS2 (40M) | 2 | 20 | ~1M | ~0.166 | 100% as Red | 0% | 50/50 | Equal to v1 direct, ties minimax d3 as Red |
| **v2-f** ✓ | DS2 (55M) | 5 | 20 | ~400K | ~0.09 | 100% as Red | 100% as Red | 50/50 | **Best v2 so far** — matches v1 vs minimax |
| v2-g | DS2 (55M) | 7 | 14 | ~150K | ~0.09 | 100% as Red | 0% | 0% | Overfit — good loss but poor gameplay, too few samples |
| v2-h | DS2 (55M) | 6 | 12 | ~200K | ~0.098 | — | 0% | 50/50 | Same pattern as min_visits=7 — narrow coverage |
| v2-i | DS2 (55M) | 4 | 12 | 528K | — | — | 0% | 0W 5L 5D | New: draws as Yellow vs v1 — very defensive play, avg 39 moves |
| v2-j | DS2 (55M) | 4 | 8 | 528K | — | — | 0% | 0% | Underfit — too few epochs for 528K samples |
| v2-k | DS2 (55M) | 5 | 8 | ~400K | — | 0% | — | — | Underfit — avg game 34.5 moves, defensive but no wins |
| v2-l | DS2 (55M) | 5 | 14 | ~400K | — | 0% | — | — | Underfit still |
| v2-m | DS2 (55M) | 5 | 24 | ~400K | — | 100% as Red | 0W 5L 5D | — | Slight overfit vs 20ep — draws as Red instead of wins, avg 38.5 move games |

---

## Key Findings

### On min_visits
- **min_visits=1**: useless — Q-values from single rollouts are noise. Loss ~0.67, untrainable.
- **min_visits=2**: broad coverage but noisy. Works on DS1 (v1) but not DS2 — suggests DS1 has higher average MCTS depth per position.
- **min_visits=3–5**: sweet spot for DS2. More coverage than 7+, enough quality signal.
- **min_visits=5 on 55M**: best results — enough records pass the filter (~400K) with enough diversity.
- **min_visits=7**: val_loss looks good (~0.09) but gameplay collapses — classic overfitting with too few samples (~150K).

### On epochs
- Too few → underfit, loss hasn't converged yet.
- Too many → overfit. In the min_visits=5, 20ep run, val_loss started ticking up around epoch 18–20, suggesting ~17 epochs is the real optimum.
- **Rule of thumb**: stop when val_loss hasn't improved for 3 consecutive epochs.

### On dataset quality
- **DS1 vs DS2**: DS1 (23M records) produces stronger models despite fewer records. DS2 records have median visits=1 even at 55M total, meaning most positions were barely explored by MCTS. DS1 appears to have been generated with more simulations per position.
- **Conclusion**: more records ≠ better model. MCTS depth per position matters more than total record count.

### On val_loss vs gameplay
- Val_loss is measured on a holdout from the **same dataset distribution**. A low val_loss doesn't guarantee strong gameplay if the training distribution is narrow.
- v2-g (val_loss ~0.09, 150K samples) loses to v1 (val_loss 0.25, 590K samples) — the 590K cover a much broader range of game positions.

---

## Next Steps

- [ ] Try min_visits=6 on 55M file — between v2-f and v2-g, might find the optimum
- [ ] Keep growing DS2 — more MCTS simulations → higher visit counts → better min_visits=5+ data
- [ ] Run DS2 MCTS with more simulations per position (Rust side) to raise quality floor
- [ ] Try early stopping: watch val_loss during training, stop at epoch 17–18 for min_visits=5
