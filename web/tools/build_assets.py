"""Copy the FetchPickAndPlace-v4 scene from gymnasium-robotics into web/assets as ONE flat MJCF (includes inlined,
mesh/texture dirs rewritten), so the browser loads exactly the model PixelGrasp was trained on.
Run with the PixelGrasp venv:  python web/tools/build_assets.py
"""
import re, shutil, pathlib, json
import gymnasium_robotics

SRC = pathlib.Path(gymnasium_robotics.__file__).parent / "envs" / "assets"
OUT = pathlib.Path(__file__).resolve().parents[1] / "assets"
(OUT / "meshes").mkdir(parents=True, exist_ok=True)


def body(xml: str) -> str:
    """Inner content of a <mujoco> ... </mujoco> file."""
    return re.sub(r"^.*?<mujoco[^>]*>|</mujoco>\s*$", "", xml, flags=re.S).strip()


main = (SRC / "fetch" / "pick_and_place.xml").read_text()
for inc in ("shared.xml", "robot.xml"):
    main = main.replace(f'<include file="{inc}"></include>', body((SRC / "fetch" / inc).read_text()))
assert "<include" not in main
main = main.replace('meshdir="../stls/fetch" texturedir="../textures"', 'meshdir="meshes" texturedir="textures"')
# block.png texture is declared but never used by a material; keep it so the model compiles unchanged.
(OUT / "textures").mkdir(exist_ok=True)
shutil.copy(SRC / "textures" / "block.png", OUT / "textures" / "block.png")
meshes = sorted(set(re.findall(r'file="([^"]+\.stl)"', main)))
for m in meshes:
    shutil.copy(SRC / "stls" / "fetch" / m, OUT / "meshes" / m)
(OUT / "pick_and_place.xml").write_text(main)
json.dump({"meshes": meshes, "textures": ["block.png"]}, open(OUT / "manifest.json", "w"), indent=1)
print(f"wrote {OUT/'pick_and_place.xml'} + {len(meshes)} meshes")
