"""Export the trained PixelGrasp policy (runs/goal_k8/best.pt) to ONNX for onnxruntime-web and check parity.
Inputs: image uint8 (1,96,96,3) HWC, state float32 (1,8) = gripper xyz, 2 finger widths, goal xyz.
Outputs: pos (1,8,3) in [-1,1], grip_logit (1,8). Normalisation of `state` is inside the graph.
"""
import argparse, json, pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pixelgrasp.policy import BCPolicyRunner

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default=str(ROOT / "web" / "models" / "pixelgrasp_k8.onnx"))
a = ap.parse_args()
r = BCPolicyRunner(a.ckpt, device="cpu"); m = r.model.float().eval()
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
g = torch.Generator().manual_seed(0)
img = torch.randint(0, 256, (1, 96, 96, 3), dtype=torch.uint8, generator=g); st = torch.randn(1, 8, generator=g)
torch.onnx.export(m, (img, st), a.out, input_names=["image", "state"], output_names=["pos", "grip_logit"], opset_version=17, dynamo=False)
import onnxruntime as ort
s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
with torch.no_grad(): ref = m(img, st)
o = s.run(None, {"image": img.numpy(), "state": st.numpy()})
d = (float(np.abs(o[0] - ref[0].numpy()).max()), float(np.abs(o[1] - ref[1].numpy()).max()))
meta = {"chunk": r.chunk, "exec_steps": r.exec_steps, "use_goal": r.use_goal, "parity_max_abs": d,
        "ckpt": pathlib.Path(a.ckpt).as_posix().split("/runs/")[-1], "bytes": pathlib.Path(a.out).stat().st_size}
json.dump(meta, open(pathlib.Path(a.out).with_suffix(".json"), "w"), indent=1)
print(meta); assert max(d) < 1e-3
