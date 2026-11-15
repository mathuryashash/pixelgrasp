# PixelGrasp — review brief for the council (v1, 16 Nov 2026)

Read fully. Do NOT modify any files or branches. Repo: `D:/pixelgrasp` (also https://github.com/mathuryashash/pixelgrasp).
Python: `D:/pixelgrasp/.venv/Scripts/python.exe` (run from `D:/pixelgrasp`). Tests: `python -m pytest tests -q` (8 tests, ~30 s).

## What this is
A 2-week "skill-gap sprint" by a final-year Indian ISE student (graduating July 2027) to earn ONE honest resume line in
robotics/perception for a "Physical AI intern" JD (Cyient) that lists "computer vision, deep learning, robotics simulation
(Isaac Sim / Gazebo / MuJoCo)". He has no prior robotics work. The repo must let a stranger re-run every number.

Deliverable: a camera-only policy (RGB 96x96 from a fixed third-person camera + 5-D proprio: gripper xyz + finger widths)
for `FetchPickAndPlace-v4` (Gymnasium-Robotics / MuJoCo 4.0.1), trained by behaviour cloning from a scripted expert.

## Protocol (fixed before any policy was trained)
- Eval seeds 0..99 (frozen). Dev seeds 5000+ (checkpoint selection). Demo seeds 10000+ (training). Disjoint by construction.
- Success = `info["is_success"]` at the final step (env's own 5 cm threshold), 50-step horizon.
- Baselines on the eval set (results/baseline_*.json): expert (privileged state) 100/100 [CI 96.3–100]; zero-action 2/100;
  uniform-random 2/100 (the 2 are seeds where the cube spawns on the goal).
- Failure taxonomy per episode: never_reached / never_grasped / dropped / misplaced (pixelgrasp/env.py classify_failure).
- Data: 5,000 expert episodes, 105,343 steps, 0 rejected; open-loop replay of recorded actions reproduces success 30/30;
  no frozen observations; cube visible in 100% of frames, goal marker in 80% (occluded by arm otherwise).
- Published reference points for the same env/method are in RESEARCH.md (state-BC 200 demos 0.81; image-BC 0.80 with
  200k eps; DAgger 0.89–0.96; action chunking is the main lever per ACT).

## Files
pixelgrasp/env.py (camera, proprio, expert, taxonomy) · eval.py (harness, Wilson CI, video) · collect.py (multiproc demos)
· check_data.py (replay + frozen-obs + pixel checks) · policy.py (ResNet-18 + proprio -> K-step chunk; grip = sigmoid head)
· train.py (AdamW, OneCycle, bf16, random-shift aug, dev-success checkpointing) · tests/test_core.py · PLAN.md · RESEARCH.md.

## Current run
runs/bc_k1_2k: chunk 1, 2,000 train eps, 15 epochs, augment. ~55 s/epoch on an RTX 4060 Laptop. Results pending.
Planned ablation rows: k=1 vs k=8 (exec 4); 500 / 2,000 / 4,900 eps; 3 seeds on the best config; camera-shift robustness.

## What the council should judge
Return `APPROVE | CHANGES_REQUIRED` + numbered findings tagged BLOCKER / SHOULD / NIT, each with the exact change, < 900 words.
