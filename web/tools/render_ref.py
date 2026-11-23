"""Dump what MuJoCo's own renderer draws (mjvScene) for the policy camera, so the three.js renderer can match it:
per-object final rgba/specular/shininess/emission, the light set, camera, and sample pixel stats of reference renders.
Writes web/assets/render_ref.json.
"""
import json, pathlib, sys
import numpy as np, mujoco, imageio
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pixelgrasp.env import make_env, CAMERA

env = make_env(); u = env.unwrapped
obs, _ = env.reset(seed=0); img = env.render()
m, d = u.model, u.data
opt = mujoco.MjvOption(); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]; cam.lookat[:] = CAMERA["lookat"]
scn = mujoco.MjvScene(m, maxgeom=1000)
mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
geoms = []
for i in range(scn.ngeom):
    g = scn.geoms[i]
    geoms.append({"objtype": int(g.objtype), "objid": int(g.objid), "category": int(g.category), "type": int(g.type),
                  "dataid": int(g.dataid), "size": [float(x) for x in g.size], "rgba": [float(x) for x in g.rgba],
                  "emission": float(g.emission), "specular": float(g.specular), "shininess": float(g.shininess),
                  "reflectance": float(g.reflectance), "transparent": int(g.transparent),
                  "name": mujoco.mj_id2name(m, int(g.objtype), int(g.objid))})
lights = []
for i in range(scn.nlight):
    L = scn.lights[i]
    lights.append({"pos": list(map(float, L.pos)), "dir": list(map(float, L.dir)), "type": int(L.type), "headlight": int(L.headlight),
                   "ambient": list(map(float, L.ambient)), "diffuse": list(map(float, L.diffuse)), "specular": list(map(float, L.specular)),
                   "castshadow": int(L.castshadow)})
glcam = scn.camera[0]
out = {"geoms": geoms, "lights": lights,
       "camera": {"pos": list(map(float, glcam.pos)), "forward": list(map(float, glcam.forward)), "up": list(map(float, glcam.up)),
                  "frustum_center": float(glcam.frustum_center), "frustum_width": float(glcam.frustum_width),
                  "frustum_bottom": float(glcam.frustum_bottom), "frustum_top": float(glcam.frustum_top),
                  "frustum_near": float(glcam.frustum_near), "frustum_far": float(glcam.frustum_far)},
       "fovy": float(m.vis.global_.fovy), "skybox": {"rgb1": [0.44, 0.85, 0.56], "rgb2": [0.46, 0.87, 0.58]},
       "flags": {"shadow": int(scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]), "fog": int(scn.flags[mujoco.mjtRndFlag.mjRND_FOG]),
                 "haze": int(scn.flags[mujoco.mjtRndFlag.mjRND_HAZE]), "skybox": int(scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX])},
       "img_mean_rgb": img.reshape(-1, 3).mean(0).tolist(), "corner_px": img[0, 0].tolist(), "center_px": img[48, 48].tolist()}
json.dump(out, open(ROOT / "web" / "assets" / "render_ref.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "geoms"}, indent=1))
for g in geoms:
    if g["rgba"][3] > 0: print(g["name"], g["category"], g["type"], [round(x, 3) for x in g["rgba"]], g["specular"], g["shininess"], g["emission"], g["transparent"])
