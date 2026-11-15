"""PixelGrasp environment helpers: fixed camera, observation packing, scripted expert, open-loop replay check.

The policy sees ONLY: an RGB image from a fixed third-person camera + proprioception (gripper xyz, finger widths).
The scripted expert additionally reads the cube and goal positions (privileged state). That expert is the
demonstrator and the upper bound; its privileged inputs never reach the vision policy.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPickAndPlace-v4"
IMG = 96
MAX_STEPS = 50
# Chosen by scratch/cam_probe.py: cube + goal + gripper visible on 12/12 seeds at 96x96.
CAMERA = {"distance": 0.9, "azimuth": 180.0, "elevation": -35.0, "lookat": [1.33, 0.75, 0.55]}
PROPRIO_DIM = 5          # grip xyz + 2 finger widths
ACTION_DIM = 4           # dx dy dz (scaled by env) + gripper (+1 open / -1 close)


def make_env(seed_render: bool = True, img: int = IMG, camera: dict | None = None):
    cam = dict(CAMERA if camera is None else camera)
    return gym.make(ENV_ID, render_mode="rgb_array" if seed_render else None, width=img, height=img,
                    default_camera_config=cam, max_episode_steps=MAX_STEPS)


def proprio(obs: dict) -> np.ndarray:
    o = obs["observation"]
    return np.concatenate([o[0:3], o[9:11]]).astype(np.float32)


def privileged(obs: dict) -> dict:
    o = obs["observation"]
    return {"grip": o[0:3], "obj": o[3:6], "fingers": o[9:11], "goal": obs["desired_goal"]}


class Expert:
    """Phase machine on privileged state: hover -> descend -> close -> carry. 100% on 20 seeds (expert_probe.py)."""

    def __init__(self, gain: float = 10.0):
        self.gain, self.phase = gain, 0

    def reset(self):
        self.phase = 0

    def act(self, obs: dict) -> np.ndarray:
        p = privileged(obs); grip, obj, goal = p["grip"], p["obj"], p["goal"]
        if self.phase == 0:
            target, g = obj + np.array([0, 0, 0.06]), 1.0
            if np.linalg.norm(target - grip) < 0.01: self.phase = 1
        if self.phase == 1:
            target, g = obj + np.array([0, 0, -0.005]), 1.0
            if np.linalg.norm(target - grip) < 0.008: self.phase = 2
        if self.phase == 2:
            target, g = grip, -1.0
            if p["fingers"].sum() < 0.052: self.phase = 3
        if self.phase == 3:
            target, g = goal, -1.0
        a = np.clip(self.gain * (target - grip), -1, 1)
        return np.array([*a, g], dtype=np.float32)


def classify_failure(info_last: dict, obj_traj: np.ndarray, grip_traj: np.ndarray, goal: np.ndarray) -> str:
    """Coarse failure taxonomy from the trajectory. 'success' if the env said so at the final step."""
    if info_last.get("is_success", 0) >= 1: return "success"
    lifted = (obj_traj[:, 2] > obj_traj[0, 2] + 0.02).any()
    near_obj = (np.linalg.norm(obj_traj - grip_traj, axis=1) < 0.03).any()
    if not near_obj: return "never_reached"
    if not lifted: return "never_grasped"
    final_d = np.linalg.norm(obj_traj[-1] - goal)
    if obj_traj[-1, 2] < obj_traj[0, 2] + 0.01 and final_d > 0.05: return "dropped"
    return "misplaced"
