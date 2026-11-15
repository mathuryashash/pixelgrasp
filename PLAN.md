# Robotics sprint — vision pick-and-place in MuJoCo (2 weeks)

Goal: one public repo that proves "computer vision + perception + deep learning + robotics simulation" for the
Cyient Physical AI intern JD (and any similar one), with a metric and a demo video. Then apply via RVITM P&T.

Working dir: `D:/robotics-sprint` (venv `.venv`, Python 3.11, mujoco 4.0.1, gymnasium-robotics, torch cu128).

## The deliverable (what the resume line will say)
> Trained a camera-only policy (ResNet-18 → 4-DoF action) for Fetch pick-and-place in MuJoCo via behaviour cloning
> from 2,000 scripted demonstrations; X% success on 100 held-out randomized scenes (expert 100%), domain-randomized
> lighting/texture, with an eval harness and per-failure breakdown.

Every number in that line is produced by a script in the repo (`eval.py`) and recorded in `RESULTS.md`.

## Why this shape
- MuJoCo is on the JD's tool list (Isaac Sim, Gazebo, MuJoCo). pip install, runs on the RTX 4060, no ROS.
- Pick-and-place is the canonical manipulation task; a vision policy = perception + DL + control in one artefact.
- Scripted expert already hits 100% (`expert_probe.py`, 20/20), so data is free and the success metric is honest.
- Day-0 checks done: offscreen render 320x240 OK; FetchPickAndPlace-v4 obs/action verified; expert 100%.

## Plan (10 working days)
Day 1  Repo skeleton (`envs/`, `data/`, `models/`, `eval.py`, `collect.py`, `train.py`), README stub, MIT licence.
       `collect.py`: run expert, save (image 128x128, proprio[grip pos, gripper state], action) per step; 2,000 episodes
       ≈ 45k samples → .npz shards. Target ≤ 30 min on CPU (render is the cost; use 96x96 if slow).
Day 2  `eval.py`: N seeds, success rate, mean steps, failure taxonomy (never grasped / dropped / wrong place), CSV + JSON.
       Run it on the expert → baseline row in RESULTS.md. Record a 10-s expert video (imageio, mp4).
Day 3  `train.py`: ResNet-18 (torchvision, ImageNet init) on image + proprio → 4-D action, MSE/L1, AdamW, 20 epochs,
       mixed precision. Log loss curves (CSV; optional W&B). First training run overnight.
Day 4  Closed-loop eval of the BC policy (100 seeds). Expect 30-60% first try. Write down what fails.
Day 5  Fixes that usually matter: action chunking (predict 4 steps), gripper as a separate classification head,
       random-crop augmentation, more data (5k eps). Re-eval. Keep a results table with one row per change.
Day 6  Domain randomisation: lighting, table/cube colour, camera jitter (MuJoCo model edits via `mjcf`). Train with DR,
       eval on DR and on clean → generalisation row.
Day 7  Buffer / second training run / seed variance (3 seeds, report mean ± std).
Day 8  Demo video: side-by-side camera view + 3rd-person, 4 episodes, success overlays. README with GIF.
Day 9  Write-up: RESULTS.md (table, failure modes, what didn't work), ARCHITECTURE.md (one diagram), VERIFY.md (exact
       commands + expected output), tests (`pytest`: env resets, dataset shapes, policy forward pass).
Day 10 Resume v7 variant (robotics), P&T e-mail to apply for Cyient, LinkedIn post about the repo.

## Guard-rails
- Report the real success rate, whatever it is. 45% with a clear failure analysis beats a vague "works".
- No ROS claim, no Isaac Sim claim on the resume unless actually used. "MuJoCo + Gymnasium-Robotics" only.
- Every metric in README traces to a command in VERIFY.md.
- If BC stalls below ~30% by Day 5, fall back to a state-based BC policy as an upper-bound row and keep the vision
  result honest; do not fake it with privileged info.

## Stretch (only if Days 1-7 finish early)
- DAgger loop (expert relabels policy rollouts) — usually +15-25 pts success.
- Isaac Lab / ROS 2 bridge in WSL2 (he said WSL2 is OK) — only as a separate branch.

## Commands so far
```
cd D:/robotics-sprint && source .venv/Scripts/activate
python expert_probe.py 20      # SUCCESS 20/20 = 100% | mean steps 21.9
```
