"""Success map of the trained policy over cube xy offsets (grid, 5 cm step, +-25 cm) from the home pose.
3 random goals per cell (training goal distribution). Writes scratch/range_grid.json and prints a text map."""
import json, sys
import numpy as np, gymnasium as gym
sys.path.insert(0, r"D:\pixelgrasp")
from pixelgrasp.env import CAMERA, IMG
from pixelgrasp.policy import BCPolicyRunner

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", width=IMG, height=IMG, default_camera_config=dict(CAMERA), max_episode_steps=60)
pol = BCPolicyRunner(r"D:\pixelgrasp\runs\goal_k8\best.pt", device="cuda")
steps = np.round(np.arange(-0.25, 0.2501, 0.05), 3)
rng = np.random.default_rng(11); grid = {}
for dx in steps:
    for dy in steps:
        if max(abs(dx), abs(dy)) < 0.149 or np.hypot(dx, dy) < 0.1: continue
        ok = 0
        for k in range(K):
            obs, _ = env.reset(seed=int(rng.integers(1e9))); u = env.unwrapped; g = u.initial_gripper_xpos[:2]
            q = u._utils.get_joint_qpos(u.model, u.data, "object0:joint").copy(); q[:2] = g + [dx, dy]
            u._utils.set_joint_qpos(u.model, u.data, "object0:joint", q); u._mujoco.mj_forward(u.model, u.data)
            obs = u._get_obs(); pol.reset()
            for t in range(50):
                obs, r, term, trunc, info = env.step(pol.act(obs, env.render()))
            ok += int(info["is_success"] >= 1)
        grid[f"{dx:+.2f},{dy:+.2f}"] = ok
json.dump({"K": K, "grid": grid}, open(r"D:\pixelgrasp\scratch\range_grid.json", "w"), indent=1)
print("rows = dx (toward/away from robot base: - is closer to robot), cols = dy; '.' = inside training square")
print("       " + " ".join(f"{y:+.2f}" for y in steps))
for dx in steps:
    print(f"{dx:+.2f}  " + " ".join(f"  {grid.get(f'{dx:+.2f},{dy:+.2f}', '.')!s:>2} " for dy in steps))
