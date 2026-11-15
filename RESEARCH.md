# Reference points (found 15 Nov 2026) — what others got on this exact task shape

Same environment (FetchPickAndPlace-v4, Gymnasium-Robotics / MuJoCo):
- hhmm1122/fetch-imitation-learning (HF + GitHub): state-based BC from a SAC+HER expert, 100-seed eval harness.
  BC 50 demos 0.31 ± 0.05; BC 200 demos 0.81 ± 0.06; DAgger 50 demos + 4k expert queries 0.89 ± 0.02; expert 1.00.
  Honest framing ("engineering + analysis project, not novel research") — copy that tone.
- Ducret, MVA project report (RGB-D 224x224 images, same env): BC 80% ± 4, DAgger 96% ± 2, DART 78%; 500 eval
  episodes; ~200k episodes seen, ~12 h training. Shows image BC on this task CAN reach ~80% with enough data.

Same method, other MuJoCo arms:
- ansh1113/diffusion-pick (Panda, 200 scripted IK demos, RGB wrist+front, ResNet-18): Diffusion Policy 87% → 95% (v2
  with recovery demos), ACT 76%; robustness splits for distractors / lighting / camera shift; 100 test seeds; ONNX export.
  Good template for the eval matrix (nominal / lighting / camera-shift columns).
- andomeder/act-mujoco-manipulation: ACT (ResNet-18 + transformer, chunk 100) in MuJoCo, 2,000 epochs ≈ 3–4 h on an
  Intel Arc. Hydra configs + W&B + video eval pipeline.
- Chase Bonfiglio SO-101: ACT ResNet-18 (~51.6 M params), 30k steps in 3 h 03 min on a GTX 1080 Ti, peak 3.9 GB VRAM.
  => an RTX 4060 8 GB is more than enough.
- LeRobot docs: ACT trains in ~1 h on an RTX 3060; small datasets (tens–hundreds of episodes) suffice.

Method lessons:
- ACT paper (Zhao et al. 2023): action chunking is the key trick — avg success 1% at k=1 → 44% at k=100 with ensembling
  off; adding chunking to plain BC-ConvMLP also helps. Scripted-data policies beat human-data ones in sim.
- Robomimic PickPlaceCan study (fmarcantoni): plain BC 0.04; image BC 0.20; RNN-BC seq-20 0.90 peak / 0.80 final;
  Diffusion 0.90. Temporal context or chunking matters more than the backbone.
- Eugene Frizza blog (MLP + ResNet-18 BC from scripted demos, xArm): failed to generalise to randomised block
  positions — covariate shift, policy freezes in unseen states. Lost a week to a bug where observations were static:
  ALWAYS (1) open-loop replay recorded actions and confirm the grasp reproduces, (2) plot recorded trajectories,
  before training anything.

Implications for this sprint (PLAN.md):
- Baseline: single-step ResNet-18 BC will probably land 30–60% on 100 held-out seeds. Report it.
- First upgrade: action chunking (k=4–8) + more demos (2k → 5k). Expect a big jump; cite ACT for why.
- Second upgrade if time: DAgger loop (expert relabels policy rollouts) — the one method that reliably closes the gap
  on this exact task (0.81 → 0.89 state-based; 80% → 96% image-based).
- Eval protocol: 100 fixed seeds, success = info["is_success"] at the final step, 3 training seeds, mean ± std.
  Robustness columns: nominal / lighting / camera jitter.
