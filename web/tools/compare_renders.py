"""Compare browser renders (test/browser_renders.json, base64 96x96x3) with MuJoCo's renders of the same layouts.
Writes test/compare.png (top: MuJoCo, middle: browser, bottom: |diff| x4) and prints per-image stats."""
import json, base64, pathlib, sys
import numpy as np, imageio
T = pathlib.Path(__file__).resolve().parents[1] / "test"
b = json.load(open(T / "browser_renders.json"))
rows = [[], [], []]
for s in sorted(b, key=int):
    web = np.frombuffer(base64.b64decode(b[s]), np.uint8).reshape(96, 96, 3).astype(int)
    mj = imageio.imread(T / "ref_renders" / f"seed{s}.png")[..., :3].astype(int)
    d = np.abs(web - mj)
    print(f"seed{s}: mean|diff| {d.mean():5.2f}  p95 {np.percentile(d, 95):5.1f}  mean rgb web {web.reshape(-1,3).mean(0).round(1)} mj {mj.reshape(-1,3).mean(0).round(1)}")
    rows[0].append(mj); rows[1].append(web); rows[2].append(np.clip(d * 4, 0, 255))
img = np.concatenate([np.concatenate(r, 1) for r in rows], 0).astype(np.uint8)
img = img.repeat(3, 0).repeat(3, 1)
imageio.imwrite(T / "compare.png", img)
print("wrote", T / "compare.png")
