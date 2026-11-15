"""Scripted expert for FetchPickAndPlace-v4 (privileged state). Measures success rate over N seeds.
obs['observation'] layout: [0:3] grip_pos, [3:6] object_pos, [6:9] object_rel_pos, [9:11] gripper_state, ...
action: dx, dy, dz (scaled 0.05 m/step inside env), gripper (+1 open, -1 close)."""
import sys, numpy as np, gymnasium as gym, gymnasium_robotics
gym.register_envs(gymnasium_robotics)

def expert_action(obs, phase, gain=10.0):
    o = obs["observation"]; grip, obj, goal = o[0:3], o[3:6], obs["desired_goal"]
    if phase == 0:                                # hover above object
        target = obj + np.array([0, 0, 0.06]); g = 1.0
        if np.linalg.norm(target - grip) < 0.01: phase = 1
    if phase == 1:                                # descend
        target = obj + np.array([0, 0, -0.005]); g = 1.0
        if np.linalg.norm(target - grip) < 0.008: phase = 2
    if phase == 2:                                # close
        target = grip; g = -1.0
        if o[9] + o[10] < 0.052: phase = 3        # fingers nearly closed on the 5 cm cube
    if phase == 3:                                # carry to goal
        target = goal; g = -1.0
    a = np.clip(gain * (target - grip), -1, 1)
    return np.array([*a, g], dtype=np.float32), phase

def run(n=20, render=False, seed0=0):
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array" if render else None, width=128, height=128)
    succ, lens = 0, []
    for ep in range(n):
        obs, info = env.reset(seed=seed0 + ep); phase = 0
        for t in range(50):
            a, phase = expert_action(obs, phase)
            obs, r, term, trunc, info = env.step(a)
            if render: env.render()
            if info["is_success"] and phase == 3 and t > 5: break
        ok = bool(info["is_success"]); succ += ok; lens.append(t + 1)
        print(f"ep {ep:02d} seed {seed0+ep} success={int(ok)} steps={t+1} final_phase={phase} dist={np.linalg.norm(obs['achieved_goal']-obs['desired_goal']):.3f}")
    print(f"SUCCESS {succ}/{n} = {succ/n:.0%} | mean steps {np.mean(lens):.1f}")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
