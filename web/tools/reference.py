"""Reference data for the browser port (run with the PixelGrasp venv from the repo root):
  1. checks the flattened web/assets MJCF compiles to the same model as gymnasium-robotics' file;
  2. records scripted-expert episodes from the real gym env (layout + actions + per-step gripper/cube positions)
     so web/test/parity.mjs can replay the same actions in MuJoCo WASM and compare;
  3. writes the eval layouts (cube xy, goal xyz) used by the browser eval.
"""
import json, pathlib, sys
import numpy as np, mujoco
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pixelgrasp.env import make_env, Expert, privileged, CAMERA, IMG

OUT = ROOT / "web" / "assets"
env = make_env(seed_render=True)
u = env.unwrapped
flat = mujoco.MjModel.from_xml_path(str(OUT / "pick_and_place.xml"))
for k in ("nq", "nv", "nbody", "ngeom", "nsite", "nmesh", "neq", "nu", "nmocap"):
    assert getattr(flat, k) == getattr(u.model, k), k
print("flat model matches gym model")

def layout(obs):
    return {"cube": [float(x) for x in privileged(obs)["obj"]], "goal": [float(x) for x in obs["desired_goal"]]}

ref = {"initial_qpos": u.initial_qpos.tolist(), "initial_qvel": u.initial_qvel.tolist(),
       "initial_gripper_xpos": u.initial_gripper_xpos.tolist(), "height_offset": float(u.height_offset),
       "camera": CAMERA, "img": IMG, "episodes": []}
for seed in range(5):
    obs, _ = env.reset(seed=1000 + seed); e = Expert(); e.reset()
    ep = {"seed": 1000 + seed, **layout(obs), "obj_qpos": u._utils.get_joint_qpos(u.model, u.data, "object0:joint").tolist(),
          "actions": [], "grip": [], "obj": [], "fingers": [], "success": []}
    for t in range(50):
        a = e.act(obs); obs, r, term, trunc, info = env.step(a)
        p = privileged(obs)
        ep["actions"].append(a.tolist()); ep["grip"].append(p["grip"].tolist()); ep["obj"].append(p["obj"].tolist())
        ep["fingers"].append(p["fingers"].tolist()); ep["success"].append(int(info["is_success"]))
    ref["episodes"].append(ep)
json.dump(ref, open(OUT / "reference.json", "w"))
# eval layouts: seeds 0..299 are the frozen eval set in the repo; the browser eval uses the first 50 of them.
lay = []
for seed in range(50):
    obs, _ = env.reset(seed=seed); lay.append({"seed": seed, **layout(obs)})
json.dump(lay, open(OUT / "eval_layouts.json", "w"))
# one reference render per layout pose (start of episode) for the renderer comparison
import imageio
(ROOT / "web" / "test" / "ref_renders").mkdir(parents=True, exist_ok=True)
for seed in range(6):
    obs, _ = env.reset(seed=seed)
    imageio.imwrite(ROOT / "web" / "test" / "ref_renders" / f"seed{seed}.png", env.render())
print("success per ref episode:", [ep["success"][-1] for ep in ref["episodes"]])
