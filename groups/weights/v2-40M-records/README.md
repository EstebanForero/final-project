# v2 — 40M records

| Field | Value |
|---|---|
| q_values records | ~39,929,774 total / 293,645 with min_visits ≥ 5 |
| q_values size | 952 MB |
| q_values source | `training/q_values` |
| trained | 2026-05-22 |
| min_visits | 5 |
| epochs | 20 |
| weights | `connect4_qnet.bin` / `connect4_qnet.safetensors` |
| final train_loss | 0.099783 |
| final val_loss | 0.103478 |

`q_values` is a symlink to the source file — do not delete the original.

## Loss curve

| Epoch | Train loss | Val loss |
|---|---|---|
| 1  | 0.181039 | 0.157507 |
| 2  | 0.146936 | 0.152983 |
| 3  | 0.135272 | 0.130963 |
| 4  | 0.128493 | 0.123644 |
| 5  | 0.123129 | 0.118774 |
| 6  | 0.118755 | 0.115669 |
| 7  | 0.115975 | 0.114239 |
| 8  | 0.112885 | 0.110901 |
| 9  | 0.110717 | 0.109006 |
| 10 | 0.108596 | 0.109420 |
| 11 | 0.107408 | 0.105976 |
| 12 | 0.106157 | 0.104776 |
| 13 | 0.104471 | 0.105253 |
| 14 | 0.103946 | 0.105452 |
| 15 | 0.102954 | 0.102589 |
| 16 | 0.102065 | 0.103192 |
| 17 | 0.101356 | 0.101385 |
| 18 | 0.100959 | 0.102372 |
| 19 | 0.100056 | 0.100776 |
| 20 | 0.099783 | 0.103478 |
