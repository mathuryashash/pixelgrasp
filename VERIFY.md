# Verify every number

Every figure in README.md and RESULTS.md is produced by a command below. Timings are for one RTX 4060 Laptop (8 GB), 16 GB RAM.
Install first: [INSTALL.md](INSTALL.md). Run everything from the repo root with the venv active.

## 0. Tests (40 s)

```bash
python -m pytest tests -q            # 14 passed
```

## 1. Baselines on the frozen eval seeds (≈ 2 min each)

```bash
python -m pixelgrasp.eval --policy expert --seeds 300 --out results/baseline_expert.json   # 300/300
python -m pixelgrasp.eval --policy zero   --seeds 300 --out results/baseline_zero.json     # 11/300
python -m pixelgrasp.eval --policy random --seeds 300 --out results/baseline_random.json   # 11/300
```

## 2. Demonstrations (≈ 5 min, 10 workers; 250,000 steps, ~7 GB uncompressed in RAM when training on all of it)

```bash
python -m pixelgrasp.collect --episodes 5000 --out data/demos --workers 10 --shard 100
python -m pixelgrasp.check_data data/demos --episodes 15      # replay_success 15, frozen_obs_episodes 0
```

`meta.json` records `seed_base: 10000`, `episodes_rejected: 0`.

## 3. Train the main policy (≈ 45 min)

```bash
python -m pixelgrasp.train --data data/demos --out runs/goal_k8 --chunk 8 --exec-steps 4 --epochs 12 \
       --max-train-eps 2000 --val-eps 100 --eval-every 4 --eval-seeds 100 --save-every 1 --augment --goal
```

`runs/goal_k8/log.json` ends with `"dev_success": 1.0`. `best.pt` is the checkpoint with the best dev success.

## 4. Evaluate it (≈ 3 min)

```bash
python -m pixelgrasp.eval --policy bc --ckpt runs/goal_k8/best.pt --seeds 300 --out results/goal_k8.json
# "success_rate": 0.9967, "ci95": [0.981, 0.999], "causes": {"success": 299, "dropped": 1}
```

## 5. Ablations and robustness

```bash
# single-step policy
python -m pixelgrasp.train --data data/demos --out runs/goal_k1 --chunk 1 --epochs 12 --max-train-eps 2000 --val-eps 100 --eval-every 4 --eval-seeds 100 --augment --goal
python -m pixelgrasp.eval  --policy bc --ckpt runs/goal_k1/best.pt --seeds 300 --out results/goal_k1.json
# second training seed
python -m pixelgrasp.train --data data/demos --out runs/abl_k8_seed1 --chunk 8 --exec-steps 4 --epochs 12 --max-train-eps 2000 --val-eps 100 --eval-every 4 --eval-seeds 100 --augment --goal --seed 1
python -m pixelgrasp.eval  --policy bc --ckpt runs/abl_k8_seed1/best.pt --seeds 300 --out results/abl_k8_seed1.json
# goal from pixels only (no target vector)
python -m pixelgrasp.train --data data/demos --out runs/abl_nogoal_k8 --chunk 8 --exec-steps 4 --epochs 12 --max-train-eps 2000 --val-eps 100 --eval-every 4 --eval-seeds 100 --augment
python -m pixelgrasp.eval  --policy bc --ckpt runs/abl_nogoal_k8/best.pt --seeds 300 --out results/abl_nogoal_k8.json
# 500 demos
python -m pixelgrasp.train --data data/demos --out runs/abl_500eps_k8 --chunk 8 --exec-steps 4 --epochs 12 --max-train-eps 500 --val-eps 100 --eval-every 4 --eval-seeds 100 --augment --goal
python -m pixelgrasp.eval  --policy bc --ckpt runs/abl_500eps_k8/best.pt --seeds 300 --out results/abl_500eps_k8.json
# open-loop execution of the main checkpoint
python -m pixelgrasp.eval  --policy bc --ckpt runs/goal_k8/best.pt --seeds 300 --exec-steps 8 --out results/goal_k8_openloop.json
# camera moved after training
for d in 3 6 10; do python -m pixelgrasp.eval --policy bc --ckpt runs/goal_k8/best.pt --seeds 100 --camera-shift $d --out results/goal_k8_camshift$d.json; done
```

## 6. Regenerate RESULTS.md and the videos

```bash
python scripts/make_results.py
python -m pixelgrasp.video compare  --ckpts runs/goal_k8/best.pt --labels PixelGrasp --seeds 4 9 17 --out results/compare_k8.mp4
python -m pixelgrasp.video progress --run runs/goal_k8 --epochs 1 2 4 12 --seeds 0 1 2 --out results/learning_k8.mp4
python -m pixelgrasp.video grid     --ckpt runs/goal_k8/best.pt --seeds 0-15 --out results/grid16_k8.mp4
python video/make_showcase.py --spec video/showcase_spec.json --out results/showcase.mp4
```

## What is NOT claimed

- No real robot, no ROS, no Isaac Sim. Simulation only (MuJoCo via Gymnasium-Robotics).
- The demonstrator is a scripted controller with privileged state, not a human.
- The policy receives the **target position** as input. It does **not** receive the cube position.
- Robustness to camera motion is measured, not solved (91% / 73% / 55% at 3° / 6° / 10°).
- Rollouts are deterministic given the seed; re-running gives bit-identical numbers on the same hardware/driver, and
  statistically equivalent numbers elsewhere.
