"""Video renderer for PixelGrasp: side-by-side comparisons, policy-input view, success/fail borders, episode grids.

Follows the conventions used in published imitation-learning repos (diffusion-pick, act_implementation, robomimic):
same seed across panels; high-res viewer camera beside the policy's actual 96x96 input; green/red outcome border.

Usage:
  python -m pixelgrasp.video compare --ckpts runs/bc_k1_2k/best.pt --labels "BC k=1" --seeds 0 1 2 3 --out results/compare.mp4
  python -m pixelgrasp.video progress --run runs/bc_k8_all --seeds 0 1 2 --out results/progress.mp4
  python -m pixelgrasp.video grid --ckpt runs/bc_k8_all/best.pt --seeds 0-15 --out results/grid.mp4
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
from .env import make_env, Expert, privileged, classify_failure, MAX_STEPS, IMG, CAMERA
from .eval import load_policy

VIEW = 384      # viewer resolution per panel
FONT_DIR = "C:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    for name in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rollout(policy, seed: int, view_env, pol_env):
    """Run one episode; returns list of (view_frame, policy_frame), outcome dict."""
    obs, info = pol_env.reset(seed=seed); view_env.reset(seed=seed); policy.reset()
    goal = obs["desired_goal"].copy(); obj_t, grip_t = [], []; frames = []
    for t in range(MAX_STEPS):
        img = pol_env.render(); view = view_env.render()
        frames.append((view, img))
        p = privileged(obs); obj_t.append(p["obj"].copy()); grip_t.append(p["grip"].copy())
        a = policy.act(obs, img)
        obs, r, term, trunc, info = pol_env.step(a); view_env.step(a)
        if term or trunc: break
    frames.append((view_env.render(), pol_env.render()))
    p = privileged(obs); obj_t.append(p["obj"].copy()); grip_t.append(p["grip"].copy())
    cause = classify_failure(info, np.array(obj_t), np.array(grip_t), goal)
    return frames, {"success": int(info["is_success"] >= 1), "steps": t + 1, "cause": cause}


def panel(view, pol_img, label, outcome=None, t=None, show_input=True):
    """One panel: viewer frame with label bar, optional policy-input inset, outcome border on last frames."""
    W = VIEW; bar = 44
    canvas = Image.new("RGB", (W, W + bar), (18, 18, 22))
    canvas.paste(Image.fromarray(view).resize((W, W), Image.BILINEAR), (0, bar))
    d = ImageDraw.Draw(canvas)
    d.text((10, 10), label, font=font(22, bold=True), fill=(240, 240, 240))
    if show_input:
        ins = Image.fromarray(pol_img).resize((IMG, IMG), Image.NEAREST)
        x0, y0 = W - IMG - 8, bar + W - IMG - 8
        canvas.paste(ins, (x0, y0)); d.rectangle([x0 - 1, y0 - 1, x0 + IMG, y0 + IMG], outline=(255, 255, 255), width=1)
        d.text((x0, y0 - 18), "policy input 96x96", font=font(13), fill=(230, 230, 230))
    if outcome is not None:
        col = (46, 204, 113) if outcome["success"] else (231, 76, 60)
        d.rectangle([0, bar, W - 1, W + bar - 1], outline=col, width=8)
        txt = "SUCCESS" if outcome["success"] else outcome["cause"].replace("_", " ").upper()
        d.rectangle([0, W + bar - 34, W, W + bar], fill=col)
        d.text((10, W + bar - 30), txt, font=font(20, bold=True), fill=(255, 255, 255))
    return canvas


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]


def compose_row(panels, title=None, footer=None, t=None):
    gap = 8; W = sum(p.width for p in panels) + gap * (len(panels) + 1)
    ff = font(20); tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    flines = wrap(tmp, footer, ff, W - 2 * gap) if footer else []
    th = 70 if title else 0; fh = 12 + 26 * len(flines) if flines else 0
    H = panels[0].height + gap * 2 + th + fh
    out = Image.new("RGB", (W, H), (10, 10, 12)); d = ImageDraw.Draw(out)
    if title:
        d.text((gap, 18), title, font=font(30, bold=True), fill=(255, 255, 255))
        if t is not None: d.text((W - gap - 110, 24), f"step {t:02d}/{MAX_STEPS}", font=font(20), fill=(170, 170, 170))
    x = gap
    for p in panels: out.paste(p, (x, th + gap)); x += p.width + gap
    for i, ln in enumerate(flines): d.text((gap, H - fh + 6 + 26 * i), ln, font=ff, fill=(200, 200, 200))
    return out


def hold(frames, img, n): frames.extend([np.asarray(img)] * n)


def write(path, frames, fps=10):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = [np.asarray(f) for f in frames]
    h, w = arr[0].shape[:2]; arr = [a[: h - h % 2, : w - w % 2] for a in arr]
    imageio.mimwrite(path, arr, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None)
    print("wrote", path, len(arr), "frames", f"{len(arr) / fps:.1f}s")


def parse_seeds(s):
    out = []
    for tok in s:
        m = re.match(r"^(\d+)-(\d+)$", tok)
        out.extend(range(int(m.group(1)), int(m.group(2)) + 1) if m else [int(tok)])
    return out


def cmd_compare(a):
    """Expert (privileged) vs one or more BC checkpoints on the SAME seeds, synchronized."""
    view_env, pol_env = make_env(img=VIEW), make_env()
    pols = [("Expert (sees cube position)", load_policy("expert", None))]
    for c, l in zip(a.ckpts, a.labels): pols.append((l + " (camera only)", load_policy("bc", c, a.device)))
    frames = []
    for s in parse_seeds(a.seeds):
        runs = [(lab, *rollout(p, s, view_env, pol_env)) for lab, p in pols]
        T = max(len(r[1]) for r in runs)
        for t in range(T):
            ps = []
            for lab, fr, oc in runs:
                v, pi = fr[min(t, len(fr) - 1)]
                ps.append(panel(v, pi, lab, oc if t >= len(fr) - 1 else None, t=min(t, len(fr) - 1), show_input="camera" in lab))
            frames.append(np.asarray(compose_row(ps, title=a.title or f"PixelGrasp  |  scene seed {s}", footer=a.footer, t=t)))
        hold(frames, frames[-1], 12)
    write(a.out, frames, fps=a.fps)


def cmd_progress(a):
    """Same seed, checkpoints from successive epochs side by side: 'the policy learning'."""
    cks = sorted(glob.glob(os.path.join(a.run, "ep_*.pt")))
    pick = a.epochs or [cks[0], cks[len(cks) // 3], cks[2 * len(cks) // 3], cks[-1]]
    pick = [p if os.path.exists(p) else os.path.join(a.run, f"ep_{int(p):03d}.pt") for p in pick]
    log = {r["epoch"]: r for r in json.load(open(os.path.join(a.run, "log.json")))}
    view_env, pol_env = make_env(img=VIEW), make_env()
    pols = []
    for p in pick:
        ep = int(re.search(r"ep_(\d+)", p).group(1)); r = log.get(ep, {})
        lab = f"epoch {ep}" + (f"  dev {r['dev_success']*100:.0f}%" if "dev_success" in r else "")
        pols.append((lab, load_policy("bc", p, a.device)))
    frames = []
    for s in parse_seeds(a.seeds):
        runs = [(lab, *rollout(p, s, view_env, pol_env)) for lab, p in pols]
        T = max(len(r[1]) for r in runs)
        for t in range(T):
            ps = [panel(*fr[min(t, len(fr) - 1)], lab, oc if t >= len(fr) - 1 else None, t=min(t, len(fr) - 1), show_input=False) for lab, fr, oc in runs]
            frames.append(np.asarray(compose_row(ps, title=a.title or f"Same scene (seed {s}), policy after N epochs of training", footer=a.footer, t=t)))
        hold(frames, frames[-1], 12)
    write(a.out, frames, fps=a.fps)


def cmd_grid(a):
    """N episodes of one policy in a grid, each with its outcome border. The honest 'here are 16 rollouts' shot."""
    seeds = parse_seeds(a.seeds); cols = a.cols; rows = int(np.ceil(len(seeds) / cols))
    view_env, pol_env = make_env(img=a.cell), make_env()
    pol = load_policy(a.policy, a.ckpt, a.device)
    runs = [rollout(pol, s, view_env, pol_env) for s in seeds]
    T = max(len(fr) for fr, _ in runs); frames = []
    succ = sum(oc["success"] for _, oc in runs)
    for t in range(T):
        G = Image.new("RGB", (cols * (a.cell + 6) + 6, rows * (a.cell + 6) + 6 + 60), (10, 10, 12)); d = ImageDraw.Draw(G)
        d.text((10, 14), a.title or f"{a.label}: {succ}/{len(seeds)} held-out scenes", font=font(28, bold=True), fill=(255, 255, 255))
        for i, (fr, oc) in enumerate(runs):
            v, _ = fr[min(t, len(fr) - 1)]; im = Image.fromarray(v)
            if t >= len(fr) - 1:
                ImageDraw.Draw(im).rectangle([0, 0, a.cell - 1, a.cell - 1], outline=(46, 204, 113) if oc["success"] else (231, 76, 60), width=6)
            G.paste(im, (6 + (i % cols) * (a.cell + 6), 66 + (i // cols) * (a.cell + 6)))
        frames.append(np.asarray(G))
    hold(frames, frames[-1], 15)
    write(a.out, frames, fps=a.fps)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("--ckpts", nargs="*", default=[]); c.add_argument("--labels", nargs="*", default=[])
    c.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3"]); c.add_argument("--out", required=True)
    p = sub.add_parser("progress"); p.add_argument("--run", required=True); p.add_argument("--epochs", nargs="*")
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2"]); p.add_argument("--out", required=True)
    g = sub.add_parser("grid"); g.add_argument("--policy", default="bc"); g.add_argument("--ckpt"); g.add_argument("--label", default="BC policy")
    g.add_argument("--seeds", nargs="+", default=["0-15"]); g.add_argument("--cols", type=int, default=4); g.add_argument("--cell", type=int, default=240); g.add_argument("--out", required=True)
    for s_ in (c, p, g):
        s_.add_argument("--fps", type=int, default=10); s_.add_argument("--device", default="cuda"); s_.add_argument("--title"); s_.add_argument("--footer")
    a = ap.parse_args()
    {"compare": cmd_compare, "progress": cmd_progress, "grid": cmd_grid}[a.cmd](a)


if __name__ == "__main__":
    main()
