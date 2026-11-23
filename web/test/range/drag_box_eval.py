"""Validate the web demo's drag area: uniform cube xy in DRAG box (>=10 cm from gripper), training goal distribution,
home pose start, 50 steps. Also reports the part of the box outside the training square separately."""
import json, sys
import numpy as np, gymnasium as gym
sys.path.insert(0, r"D:\pixelgrasp")
from pixelgrasp.env import CAMERA, IMG
from pixelgrasp.policy import BCPolicyRunner
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
BX, BY = (-0.20, 0.16), (-0.20, 0.20)
env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", width=IMG, height=IMG, default_camera_config=dict(CAMERA), max_episode_steps=60)
pol = BCPolicyRunner(r"D:\pixelgrasp\runs\goal_k8\best.pt", device="cuda")
rng = np.random.default_rng(2026); rows = []
while len(rows) < N:
    d = np.array([rng.uniform(*BX), rng.uniform(*BY)])
    if np.hypot(*d) < 0.1: continue
    obs, _ = env.reset(seed=int(rng.integers(1e9))); u = env.unwrapped
    q = u._utils.get_joint_qpos(u.model, u.data, "object0:joint").copy(); q[:2] = u.initial_gripper_xpos[:2] + d
    u._utils.set_joint_qpos(u.model, u.data, "object0:joint", q); u._mujoco.mj_forward(u.model, u.data)
    obs = u._get_obs(); pol.reset()
    for t in range(50): obs, r, term, trunc, info = env.step(pol.act(obs, env.render()))
    rows.append({"dx": float(d[0]), "dy": float(d[1]), "outside": bool(np.abs(d).max() > 0.15), "s": int(info["is_success"] >= 1)})
out = [r for r in rows if r["outside"]]; ins = [r for r in rows if not r["outside"]]
print(f"whole box {sum(r['s'] for r in rows)}/{len(rows)} | outside training square {sum(r['s'] for r in out)}/{len(out)} | inside {sum(r['s'] for r in ins)}/{len(ins)}")
print("fails:", [(round(r['dx'], 3), round(r['dy'], 3)) for r in rows if not r["s"]])
json.dump(rows, open(r"D:\pixelgrasp\scratch\drag_box_eval.json", "w"))
