"""Assemble PixelGrasp showcase videos from rendered clips + PIL title/result cards.

  python video/make_showcase.py --spec video/progress_spec.json --out results/progress_1116.mp4 [--vertical]

Spec: {"size": [1920,1080], "fps": 30, "segments": [
   {"card": {"title": "...", "lines": ["..", ".."], "seconds": 3}},
   {"clip": "results/compare.mp4"},
   ...]}
Cards are drawn with PIL; clips are scaled+padded to the target size; segments are joined with 0.4 s crossfades.
"""
from __future__ import annotations
import argparse, json, os, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "C:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts/truetype/dejavu"
BG = (10, 10, 12); FG = (245, 245, 245); DIM = (170, 170, 175); ACC = (46, 204, 113)


def font(size, bold=False):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        p = os.path.join(FONT_DIR, n)
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def card(spec, size):
    W, H = size; im = Image.new("RGB", size, BG); d = ImageDraw.Draw(im)
    s = min(W, H) / 1080          # same text size on 1920x1080 and 1080x1920
    y = H * (0.22 if W >= H else 0.30)
    tf = font(int(72 * s), True)
    for line in spec["title"].split("\n"):
        d.text(((W - d.textlength(line, font=tf)) / 2, y), line, font=tf, fill=FG); y += int(90 * s)
    y += int(30 * s)
    lf = font(int(40 * s))
    for line in spec.get("lines", []):
        col = ACC if line.startswith("+") else (DIM if line.startswith("-") else FG)
        txt = line[1:].strip() if line[:1] in "+-" else line
        d.text(((W - d.textlength(txt, font=lf)) / 2, y), txt, font=lf, fill=col); y += int(58 * s)
    if spec.get("footer"):
        ff = font(int(30 * s)); t = spec["footer"]
        d.text(((W - d.textlength(t, font=ff)) / 2, H - int(90 * s)), t, font=ff, fill=DIM)
    return im


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); spec = json.load(open(a.spec))
    W, H = spec.get("size", [1920, 1080]); fps = spec.get("fps", 30)
    tmp = tempfile.mkdtemp(prefix="pg_show_"); segs = []
    for i, s in enumerate(spec["segments"]):
        out = os.path.join(tmp, f"seg{i:02d}.mp4")
        if "card" in s:
            png = os.path.join(tmp, f"card{i:02d}.png"); card(s["card"], (W, H)).save(png)
            run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(s["card"].get("seconds", 3)), "-i", png,
                 "-vf", f"fps={fps},format=yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18", out])
        else:
            vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0a0c,fps={fps},format=yuv420p"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", s["clip"]]
            if s.get("trim"): cmd += ["-t", str(s["trim"])]
            run(cmd + ["-vf", vf, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", out])
        segs.append(out)
    # durations for xfade offsets
    durs = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip()) for p in segs]
    X = 0.4; fc = []; off = 0.0; prev = "[0:v]"
    for i in range(1, len(segs)):
        off += durs[i - 1] - X; lab = f"[v{i}]" if i < len(segs) - 1 else "[vout]"
        fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={X}:offset={off:.3f}{lab}"); prev = lab
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in segs: cmd += ["-i", p]
    if len(segs) > 1: cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); run(cmd)
    total = sum(durs) - X * (len(segs) - 1); print("wrote", a.out, f"{total:.1f}s")


if __name__ == "__main__":
    main()
