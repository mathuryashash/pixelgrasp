"""Paired test: same 60 layouts (cube uniform in the web drag box, goal from the env's own sampler) run in Python
(MuJoCo renderer + PyTorch) here, and written to web/test/range/drag_layouts.json for the browser bench."""
import json, sys
import numpy as np, gymnasium as gym
sys.path.insert(0, r"D:\pixelgrasp")
from pixelgrasp.env import CAMERA, IMG
from pixelgrasp.policy import BCPolicyRunner
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BX, BY = (-0.17, 0.15), (-0.17, 0.17)
env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", width=IMG, height=IMG, default_camera_config=dict(CAMERA), max_episode_steps=60)
pol = BCPolicyRunner(r"D:\pixelgrasp\runs\goal_k8\best.pt", device="cuda")
rng = np.random.default_rng(123); lay = []
while len(lay) < N:
    d = np.array([rng.uniform(*BX), rng.uniform(*BY)])
    if np.hypot(*d) < 0.1 or np.abs(d).max() <= 0.15: continue      # only the NEW part of the area
    obs, _ = env.reset(seed=int(rng.integers(1e9))); u = env.unwrapped
    cube = u.initial_gripper_xpos[:2] + d
    q = u._utils.get_joint_qpos(u.model, u.data, "object0:joint").copy(); q[:2] = cube
    u._utils.set_joint_qpos(u.model, u.data, "object0:joint", q); u._mujoco.mj_forward(u.model, u.data)
    obs = u._get_obs(); pol.reset()
    for t in range(50): obs, r, term, trunc, info = env.step(pol.act(obs, env.render()))
    lay.append({"seed": len(lay), "cube": cube.tolist(), "goal": obs["desired_goal"].tolist(), "d": d.round(3).tolist(), "py": int(info["is_success"] >= 1)})
print("python:", sum(l["py"] for l in lay), "/", N)
json.dump(lay, open(r"D:\pixelgrasp-web\web\test\range\drag_layouts.json", "w"))
