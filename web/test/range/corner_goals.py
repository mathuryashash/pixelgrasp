import sys, numpy as np, gymnasium as gym
sys.path.insert(0, r"D:\pixelgrasp")
from pixelgrasp.env import CAMERA, IMG
from pixelgrasp.policy import BCPolicyRunner
env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", width=IMG, height=IMG, default_camera_config=dict(CAMERA), max_episode_steps=60)
pol = BCPolicyRunner(r"D:\pixelgrasp\runs\goal_k8\best.pt", device="cuda")
L = [([1.152, 0.559], [1.208, 0.794, 0.757]), ([1.152, 0.939], [1.208, 0.794, 0.757]), ([1.492, 0.559], [1.208, 0.794, 0.757]),
     ([1.152, 0.559], [1.30, 0.70, 0.425]), ([1.152, 0.559], [1.40, 0.85, 0.60])]
def run(cube, goal):
    obs, _ = env.reset(seed=0); u = env.unwrapped
    q = u._utils.get_joint_qpos(u.model, u.data, "object0:joint").copy(); q[:2] = cube
    u._utils.set_joint_qpos(u.model, u.data, "object0:joint", q); u.goal = np.array(goal); u._mujoco.mj_forward(u.model, u.data)
    obs = u._get_obs(); pol.reset()
    for t in range(50): obs, r, term, trunc, info = env.step(pol.act(obs, env.render()))
    return int(info["is_success"] >= 1), round(float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])), 3)
print([run(c, g) for c, g in L])
# goal-dependence: far corners x goals across the goal box (3x3 xy x 2 heights)
g0 = env.unwrapped.initial_gripper_xpos.copy(); h = env.unwrapped.height_offset
for d in [(-0.19, -0.19), (-0.19, 0.19), (0.15, -0.19), (0.15, 0.19), (-0.19, 0.0), (0.0, 0.19)]:
    ok = n = 0
    for gx in (-0.15, 0, 0.15):
        for gy in (-0.15, 0, 0.15):
            for gz in (0, 0.33):
                s, _ = run(list(g0[:2] + d), [g0[0] + gx, g0[1] + gy, h + gz]); ok += s; n += 1
    print(d, f"{ok}/{n}", flush=True)
