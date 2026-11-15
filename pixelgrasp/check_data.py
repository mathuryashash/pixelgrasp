"""Data sanity checks (the step people skip and then lose a week on):
1. Open-loop replay: re-run the recorded actions from the same seed -> the grasp must reproduce (success).
2. Observations must change over time (no frozen-observation bug).
3. Image statistics: cube/goal pixels present.
Usage: python -m pixelgrasp.check_data data/demos --episodes 20
"""
import argparse, glob, json, os
import numpy as np
from .env import make_env, MAX_STEPS


def load(data_dir, max_episodes=None):
    """Load shards in order until max_episodes are present (the full set is ~7 GB uint8; never load it all for a check)."""
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    parts, n_eps = [], 0
    for s in shards:
        x = np.load(s); parts.append({k: x[k] for k in x.files}); n_eps += len(np.unique(x["episode"]))
        if max_episodes and n_eps >= max_episodes: break
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def replay_episode(env, seed: int, acts: np.ndarray) -> bool:
    """Open-loop replay of recorded actions. render() BEFORE every step() mirrors collection exactly:
    render() calls mj_forward, which perturbs the warm-start, so skipping it changes the physics by ~1e-4 and breaks grasps."""
    obs, info = env.reset(seed=int(seed))
    for a_ in acts:
        env.render()
        obs, r, term, trunc, info = env.step(a_)
    return bool(info["is_success"] >= 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    D = load(a.data_dir, max_episodes=a.episodes)
    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    eps = np.unique(D["episode"])[: a.episodes]
    env = make_env()
    ok = 0; frozen = 0
    for s in eps:
        m = D["episode"] == s; acts = D["action"][m][np.argsort(D["t"][m])]
        ok += int(replay_episode(env, s, acts))
        imgs = D["img"][m]
        # temporal variance: successive frames must differ (a static scene still has a moving gripper)
        if np.abs(np.diff(imgs.astype(np.int16), axis=0)).mean() < 0.05: frozen += 1
    img = D["img"]; cube = ((img < 60).all(-1)).mean(axis=(1, 2)); red = ((img[..., 0] > 180) & (img[..., 1] < 90) & (img[..., 2] < 90)).mean(axis=(1, 2))
    rep = {"episodes_checked": int(len(eps)), "replay_success": ok, "frozen_obs_episodes": frozen,
           "steps_total": meta["steps"], "episodes_total": meta["episodes_kept"], "stats_on_steps": int(len(D["t"])),
           "action_mean": D["action"].mean(0).round(3).tolist(), "action_std": D["action"].std(0).round(3).tolist(),
           "gripper_close_frac": float((D["action"][:, 3] < 0).mean()),
           "frames_with_cube_px": float((cube > 0).mean()), "frames_with_goal_px": float((red > 0).mean()),
           "proprio_mean": D["proprio"].mean(0).round(3).tolist(), "proprio_std": D["proprio"].std(0).round(3).tolist()}
    print(json.dumps(rep, indent=1))
    assert ok == len(eps), f"open-loop replay failed on {len(eps)-ok} episodes"
    assert frozen == 0, "frozen observations detected"


if __name__ == "__main__":
    main()
