"""Data sanity checks (the step people skip and then lose a week on):
1. Open-loop replay: re-run the recorded actions from the same seed -> the grasp must reproduce (success).
2. Observations must change over time (no frozen-observation bug).
3. Image statistics: cube/goal pixels present.
Usage: python -m pixelgrasp.check_data data/demos --episodes 20
"""
import argparse, glob, json, os
import numpy as np
from .env import make_env, MAX_STEPS


def load(data_dir):
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    d = [np.load(s) for s in shards]
    return {k: np.concatenate([x[k] for x in d]) for k in d[0].files}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    D = load(a.data_dir)
    eps = np.unique(D["episode"])[: a.episodes]
    env = make_env()
    ok = 0; frozen = 0
    for s in eps:
        m = D["episode"] == s; acts = D["action"][m][np.argsort(D["t"][m])]
        obs, info = env.reset(seed=int(s))
        for a_ in acts:
            env.render()          # mirror collection exactly: render() calls mj_forward, which perturbs the warm-start
            obs, r, term, trunc, info = env.step(a_)
        ok += int(info["is_success"] >= 1)
        imgs = D["img"][m]
        if imgs.std(axis=0).mean() < 1.0: frozen += 1
    img = D["img"]; cube = ((img < 60).all(-1)).mean(axis=(1, 2)); red = ((img[..., 0] > 180) & (img[..., 1] < 90) & (img[..., 2] < 90)).mean(axis=(1, 2))
    rep = {"episodes_checked": int(len(eps)), "replay_success": ok, "frozen_obs_episodes": frozen,
           "steps_total": int(len(D["t"])), "episodes_total": int(len(np.unique(D["episode"]))),
           "action_mean": D["action"].mean(0).round(3).tolist(), "action_std": D["action"].std(0).round(3).tolist(),
           "gripper_close_frac": float((D["action"][:, 3] < 0).mean()),
           "frames_with_cube_px": float((cube > 0).mean()), "frames_with_goal_px": float((red > 0).mean()),
           "proprio_mean": D["proprio"].mean(0).round(3).tolist(), "proprio_std": D["proprio"].std(0).round(3).tolist()}
    print(json.dumps(rep, indent=1))
    assert ok == len(eps), f"open-loop replay failed on {len(eps)-ok} episodes"
    assert frozen == 0, "frozen observations detected"


if __name__ == "__main__":
    main()
