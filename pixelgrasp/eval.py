"""Evaluate any policy on fixed held-out seeds. Writes JSON + CSV, optional video.

Usage:
  python -m pixelgrasp.eval --policy expert  --seeds 100 --out results/expert.json
  python -m pixelgrasp.eval --policy zero    --seeds 100 --out results/zero.json
  python -m pixelgrasp.eval --policy random  --seeds 100 --out results/random.json
  python -m pixelgrasp.eval --policy bc --ckpt runs/bc/best.pt --seeds 100 --out results/bc.json --video results/bc.mp4
Seeds 0..N-1 are the EVAL set. Training data is collected on seeds >= 10_000 so they never overlap.
"""
from __future__ import annotations
import argparse, json, time, csv, os
from collections import Counter
import numpy as np
from .env import make_env, Expert, privileged, classify_failure, MAX_STEPS, EVAL_SEED_BASE, IMG


class ZeroPolicy:
    def reset(self): pass
    def act(self, obs, img): return np.zeros(4, np.float32)


class RandomPolicy:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def act(self, obs, img): return self.rng.uniform(-1, 1, 4).astype(np.float32)


class ExpertPolicy:
    def __init__(self): self.e = Expert()
    def reset(self): self.e.reset()
    def act(self, obs, img): return self.e.act(obs)


def load_policy(name: str, ckpt: str | None, device: str = "cuda", exec_steps: int | None = None):
    if name == "expert": return ExpertPolicy()
    if name == "zero": return ZeroPolicy()
    if name == "random": return RandomPolicy()
    if name == "bc":
        from .policy import BCPolicyRunner
        return BCPolicyRunner(ckpt, device=device, exec_steps=exec_steps)
    raise ValueError(name)


def run_episode(env, policy, seed: int, record: list | None = None):
    obs, info = env.reset(seed=seed); policy.reset()
    goal = obs["desired_goal"].copy(); obj_t, grip_t = [], []
    t_inf = []
    for t in range(MAX_STEPS):
        img = env.render()
        if record is not None: record.append(img)
        p = privileged(obs); obj_t.append(p["obj"].copy()); grip_t.append(p["grip"].copy())
        t0 = time.perf_counter(); a = policy.act(obs, img); t_inf.append(time.perf_counter() - t0)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc: break
    p = privileged(obs); obj_t.append(p["obj"].copy()); grip_t.append(p["grip"].copy())
    cause = classify_failure(info, np.array(obj_t), np.array(grip_t), goal)
    return {"seed": seed, "success": int(info["is_success"] >= 1), "steps": t + 1, "cause": cause,
            "final_dist": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
            "inf_ms": float(np.mean(t_inf) * 1000)}


def evaluate(policy, n_seeds: int = 100, seed_base: int = EVAL_SEED_BASE, video: str | None = None,
             video_eps: int = 4, camera: dict | None = None, img: int | None = None):
    env = make_env(camera=camera, img=img or IMG)
    rows, frames = [], []
    for i in range(n_seeds):
        rec = frames if (video and i < video_eps) else None
        rows.append(run_episode(env, policy, seed_base + i, record=rec))
    env.close()
    succ = [r["success"] for r in rows]
    summary = {"n": n_seeds, "seed_base": seed_base, "success_rate": float(np.mean(succ)),
               "ci95": wilson(sum(succ), n_seeds), "mean_steps": float(np.mean([r["steps"] for r in rows])),
               "mean_steps_success": float(np.mean([r["steps"] for r in rows if r["success"]])) if any(succ) else None,
               "causes": dict(Counter(r["cause"] for r in rows)),
               "inf_ms": float(np.mean([r["inf_ms"] for r in rows]))}
    if video and frames:
        import imageio
        os.makedirs(os.path.dirname(video) or ".", exist_ok=True)
        imageio.mimwrite(video, frames, fps=10, macro_block_size=None)
    return summary, rows


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0: return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(float((c - h) / d), 3), round(float((c + h) / d), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["expert", "zero", "random", "bc"])
    ap.add_argument("--ckpt"); ap.add_argument("--seeds", type=int, default=100); ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", required=True); ap.add_argument("--video"); ap.add_argument("--video-eps", type=int, default=4)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--tag", default="")
    ap.add_argument("--camera-shift", type=float, default=0.0, help="deg added to azimuth AND elevation (robustness test)")
    ap.add_argument("--exec-steps", type=int, default=None, help="override how many chunk steps run before re-planning (open-loop = chunk)")
    a = ap.parse_args()
    cam = None
    if a.camera_shift:
        from .env import CAMERA
        cam = dict(CAMERA); cam["azimuth"] += a.camera_shift; cam["elevation"] += a.camera_shift
    pol = load_policy(a.policy, a.ckpt, a.device, exec_steps=a.exec_steps)
    t0 = time.time()
    summary, rows = evaluate(pol, a.seeds, a.seed_base, a.video, a.video_eps, camera=cam)
    summary.update({"policy": a.policy, "ckpt": a.ckpt, "tag": a.tag, "camera_shift": a.camera_shift, "exec_steps": getattr(pol, "exec_steps", None), "wall_s": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "episodes": rows}, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
