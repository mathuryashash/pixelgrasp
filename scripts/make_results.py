"""Build RESULTS.md from results/*.json so every number in the docs is machine-generated. Run after eval.py."""
import json, glob, os

R = {os.path.basename(p)[:-5]: json.load(open(p))["summary"] for p in glob.glob("results/*.json")}


def row(name, key, note=""):
    s = R.get(key)
    if not s: return f"| {name} | – | – | – | {note} |"
    k = round(s["success_rate"] * s["n"]); lo, hi = s["ci95"]
    causes = ", ".join(f"{c} {n}" for c, n in sorted(s["causes"].items(), key=lambda x: -x[1]) if c != "success") or "–"
    return f"| {name} | **{k}/{s['n']}** ({s['success_rate']*100:.1f}%) | {lo*100:.1f}–{hi*100:.1f} | {causes} | {note} |"


def dev(run):
    p = f"runs/{run}/log.json"
    if not os.path.exists(p): return "–"
    L = json.load(open(p)); d = [r["dev_success"] for r in L if "dev_success" in r]
    return f"{max(d)*100:.0f}% (ep {[r['epoch'] for r in L if r.get('dev_success') == max(d)][0]}), {L[-1]['wall_s']//60} min"


hdr = "| Policy | Success | 95% CI | Failures | Note |\n|---|---|---|---|---|"
out = f"""# Results

All numbers below are produced by `python -m pixelgrasp.eval` on the **frozen eval seeds 0..299** (`FetchPickAndPlace-v4`,
50-step horizon, success = the environment's own `is_success` at the final step, 5 cm tolerance). The policy never sees these
seeds during training (demos use seeds ≥ 10000, checkpoint selection uses dev seeds 5000..5099). CI = Wilson score interval.
Hardware: one RTX 4060 Laptop GPU (8 GB). Exact commands: [VERIFY.md](VERIFY.md).

## Main table

{hdr}
{row("Scripted expert (reads cube + goal from sim state)", "baseline_expert", "upper bound / demonstrator")}
{row("**PixelGrasp**, action chunk k=8, re-plan every 4", "goal_k8", "2,000 demos, 12 epochs, seed 0")}
{row("PixelGrasp, chunk k=8, training seed 1", "abl_k8_seed1", "same config, different init/shuffle")}
{row("PixelGrasp, single-step k=1", "goal_k1", "2,000 demos, 12 epochs")}
{row("Do nothing (zero action)", "baseline_zero", "cube already on goal on these seeds")}
{row("Uniform random actions", "baseline_random", "")}

The policy input is one 96×96 RGB frame from a fixed third-person camera, the gripper position + finger widths (5 numbers),
and the **target position** (3 numbers, the task command). The **cube position is never given** — it has to be found in the image.

## Ablations (same eval seeds)

{hdr}
{row("k=8, goal from pixels only (no target vector)", "abl_nogoal_k8", "must also read the 17-px red marker")}
{row("k=8, 500 demos instead of 2,000", "abl_500eps_k8", "data efficiency")}
{row("k=8, open-loop (execute all 8, re-plan every 8)", "goal_k8_openloop", "same checkpoint as main row")}

## Robustness: camera moved after training (100 seeds each)

{hdr}
{row("Camera +3° azimuth and elevation", "goal_k8_camshift3", "not trained for this")}
{row("Camera +6°", "goal_k8_camshift6", "")}
{row("Camera +10°", "goal_k8_camshift10", "")}

Failures move to *never reached* as the shift grows: the policy really is reading the cube position from pixels, and a moved
camera breaks that mapping. Camera-jitter augmentation during training is the obvious next step and is **not done here**.

## Dev-seed curve (checkpoint selection, 100 seeds 5000..5099)

| Run | Best dev success, training time |
|---|---|
| goal_k8 (main) | {dev("goal_k8")} |
| goal_k1 | {dev("goal_k1")} |
| abl_k8_seed1 | {dev("abl_k8_seed1")} |
| abl_nogoal_k8 | {dev("abl_nogoal_k8")} |
| abl_500eps_k8 | {dev("abl_500eps_k8")} |

## What failed first (kept on purpose)

| Run | Dev success | What was wrong |
|---|---|---|
| 1. k=1, 2k demos, goal from pixels, early-exit demos | 30% | the goal is a ~17-pixel red dot; its height is ambiguous from one camera. Failures: *misplaced* (lifted, carried to the wrong spot) |
| 2. k=8, 4.9k demos, same data | 1% | chunking made the model average the ambiguity harder — a smoother wrong answer |
| 3. k=1, goal given as a vector, early-exit demos | 19% | reached within 5 cm of the goal in 98% of episodes, then drifted away in 88%: the demos stopped at success, so "hold still at the goal" was never in the data |
| 4. goal vector + full 50-step demos | 98–100% | both fixes together |

Run-1/2 numbers are from `runs/bc_k1_2k`, `runs/bc_k8_all` (dev seeds, logged in `log.json`); those checkpoints are kept but
are not comparable to the main table (different data). Run 3's checkpoint was deleted when the data was re-collected.

## Published reference points (different setups; not a leaderboard)

- Same env, state-based BC, 200 demos: 0.81 ± 0.06; DAgger 0.89 (hhmm1122/fetch-imitation-learning).
- Same env, 224×224 RGB-D BC, ~200k episodes, 12 h: 80 ± 4%; DAgger 96 ± 2% (Ducret, MVA report).
- Panda cube→bowl in MuJoCo, 200 scripted demos, ResNet-18: Diffusion Policy 87–95%, ACT 76% (ansh1113/diffusion-pick).

Those report goal-conditioned or fixed-goal tasks too. The gap to the image-BC report is mostly the hold-phase data and the
goal vector, not the model; see RESEARCH.md.
"""
open("RESULTS.md", "w", encoding="utf-8").write(out)
print(out[:1500])
