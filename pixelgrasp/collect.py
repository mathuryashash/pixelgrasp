"""Collect demonstrations from the scripted expert. Saves .npz shards of (img, proprio, action, episode, t).

Usage: python -m pixelgrasp.collect --episodes 2000 --out data/demos --seed-base 10000 [--workers 8]
Seeds >= 10000 so they never overlap with eval seeds 0..999. Only SUCCESSFUL episodes are kept (success-filtered BC),
and the count of rejected episodes is recorded in meta.json.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool
from .env import make_env, Expert, proprio, goal_vec, MAX_STEPS, IMG


def collect_chunk(args):
    seeds, img_size, shard_path = args
    env = make_env(img=img_size); ex = Expert()
    imgs, props, goals, acts, eps, ts, rejected = [], [], [], [], [], [], 0
    for s in seeds:
        obs, info = env.reset(seed=s); ex.reset()
        e_imgs, e_props, e_goals, e_acts = [], [], [], []
        for t in range(MAX_STEPS):
            e_imgs.append(env.render()); e_props.append(proprio(obs)); e_goals.append(goal_vec(obs))
            a = ex.act(obs); e_acts.append(a)
            obs, r, term, trunc, info = env.step(a)
            if info["is_success"] >= 1 and ex.phase == 3 and t > 5: break   # early exit: goal reached, save the remaining budget
        if info["is_success"] < 1:
            rejected += 1; continue
        n = len(e_acts)
        imgs += e_imgs; props += e_props; goals += e_goals; acts += e_acts; eps += [s] * n; ts += list(range(n))
    env.close()
    np.savez_compressed(shard_path, img=np.stack(imgs).astype(np.uint8), proprio=np.stack(props).astype(np.float32),
                        goal=np.stack(goals).astype(np.float32),
                        action=np.stack(acts).astype(np.float32), episode=np.array(eps, np.int32), t=np.array(ts, np.int16))
    return shard_path, len(set(eps)), len(acts), rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000); ap.add_argument("--out", default="data/demos")
    ap.add_argument("--seed-base", type=int, default=10_000); ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--img", type=int, default=IMG); ap.add_argument("--shard", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seed_base, a.seed_base + a.episodes))
    chunks = [(seeds[i:i + a.shard], a.img, os.path.join(a.out, f"shard_{i // a.shard:04d}.npz")) for i in range(0, len(seeds), a.shard)]
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(collect_chunk, chunks)
    kept = sum(r[1] for r in res); steps = sum(r[2] for r in res); rej = sum(r[3] for r in res)
    meta = {"episodes_requested": a.episodes, "episodes_kept": kept, "episodes_rejected": rej, "steps": steps,
            "img": a.img, "seed_base": a.seed_base, "shards": [os.path.basename(r[0]) for r in res], "wall_s": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
