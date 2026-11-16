# Results

All numbers below are produced by `python -m pixelgrasp.eval` on the **frozen eval seeds 0..299** (`FetchPickAndPlace-v4`,
50-step horizon, success = the environment's own `is_success` at the final step, 5 cm tolerance). The policy never sees these
seeds during training (demos use seeds ≥ 10000, checkpoint selection uses dev seeds 5000..5099). CI = Wilson score interval.
Hardware: one RTX 4060 Laptop GPU (8 GB). Exact commands: [VERIFY.md](VERIFY.md).

## Main table

| Policy | Success | 95% CI | Failures | Note |
|---|---|---|---|---|
| Scripted expert (reads cube + goal from sim state) | **300/300** (100.0%) | 98.7–100.0 | – | upper bound / demonstrator |
| **PixelGrasp**, action chunk k=8, re-plan every 4 | **299/300** (99.7%) | 98.1–99.9 | dropped 1 | 2,000 demos, 12 epochs, seed 0 |
| PixelGrasp, chunk k=8, training seed 1 | **297/300** (99.0%) | 97.1–99.7 | never_grasped 2, misplaced 1 | same config, different init/shuffle |
| PixelGrasp, single-step k=1 | **297/300** (99.0%) | 97.1–99.7 | misplaced 3 | 2,000 demos, 12 epochs |
| Do nothing (zero action) | **11/300** (3.7%) | 2.1–6.4 | never_reached 289 | cube already on goal on these seeds |
| Uniform random actions | **11/300** (3.7%) | 2.1–6.4 | never_reached 283, never_grasped 5, dropped 1 |  |

The policy input is one 96×96 RGB frame from a fixed third-person camera, the gripper position + finger widths (5 numbers),
and the **target position** (3 numbers, the task command). The **cube position is never given** — it has to be found in the image.

## Ablations (same eval seeds)

| Policy | Success | 95% CI | Failures | Note |
|---|---|---|---|---|
| k=8, goal from pixels only (no target vector) | **248/300** (82.7%) | 78.0–86.5 | misplaced 32, never_grasped 19, dropped 1 | must also read the 17-px red marker |
| k=8, 500 demos instead of 2,000 | **203/300** (67.7%) | 62.2–72.7 | misplaced 74, never_grasped 19, dropped 4 | data efficiency |
| k=8, open-loop (execute all 8, re-plan every 8) | **298/300** (99.3%) | 97.6–99.8 | never_grasped 2 | same checkpoint as main row |

## Robustness: camera moved after training (100 seeds each)

| Policy | Success | 95% CI | Failures | Note |
|---|---|---|---|---|
| Camera +3° azimuth and elevation | **91/100** (91.0%) | 83.8–95.2 | never_grasped 6, dropped 3 | not trained for this |
| Camera +6° | **73/100** (73.0%) | 63.6–80.7 | never_reached 16, never_grasped 9, dropped 2 |  |
| Camera +10° | **55/100** (55.0%) | 45.2–64.4 | never_reached 27, never_grasped 13, misplaced 3, dropped 2 |  |

Failures move to *never reached* as the shift grows: the policy really is reading the cube position from pixels, and a moved
camera breaks that mapping. Camera-jitter augmentation during training is the obvious next step and is **not done here**.

## Dev-seed curve (checkpoint selection, 100 seeds 5000..5099)

| Run | Best dev success, training time |
|---|---|
| goal_k8 (main) | 100% (ep 12), 44 min |
| goal_k1 | 98% (ep 8), 37 min |
| abl_k8_seed1 | 100% (ep 12), 37 min |
| abl_nogoal_k8 | 76% (ep 12), 57 min |
| abl_500eps_k8 | 67% (ep 12), 11 min |

## What failed first (kept on purpose)

| Run | Dev success | What was wrong |
|---|---|---|
| 1. k=1, 2k demos, goal from pixels, early-exit demos | 30% (ep 5; 20% at ep 15) | the goal is a ~17-pixel red dot; its height is ambiguous from one camera. Failures: *misplaced* (lifted, carried to the wrong spot) |
| 2. k=8, 4.9k demos, same data | 1% | chunking made the model average the ambiguity harder — a smoother wrong answer |
| 3. k=1, goal given as a vector, early-exit demos | 19% | scratch diagnostic (60 episodes, checkpoint since deleted): came within 5 cm of the goal in 98%, then drifted away in 88%. The demos stopped at success, so "hold still at the goal" was never in the data |
| 4. goal vector + full 50-step demos | 98–100% | both fixes together |

Run-1/2 numbers are from `runs/bc_k1_2k`, `runs/bc_k8_all` (dev seeds, logged in `log.json`); those checkpoints are kept but
are not comparable to the main table (different data). Run 3's checkpoint was deleted when the data was re-collected.

## Published reference points (different setups; not a leaderboard)

- Same env, state-based BC, 200 demos: 0.81 ± 0.06; DAgger 0.89 (hhmm1122/fetch-imitation-learning).
- Same env, 224×224 RGB-D BC, ~200k episodes, 12 h: 80 ± 4%; DAgger 96 ± 2% (Ducret, MVA report).
- Panda cube→bowl in MuJoCo, 200 scripted demos, ResNet-18: Diffusion Policy 87–95%, ACT 76% (ansh1113/diffusion-pick).

Those report goal-conditioned or fixed-goal tasks too. The gap to the image-BC report is mostly the hold-phase data and the
goal vector, not the model; see RESEARCH.md.
