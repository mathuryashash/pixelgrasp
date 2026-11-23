// Node parity test: the JS port vs the real gymnasium-robotics env (web/assets/reference.json, written by tools/reference.py).
// 1) initial state after _env_setup matches; 2) replaying the Python expert's ACTIONS gives the same trajectory;
// 3) the JS expert, run closed-loop, solves the same seeds.     Run: node test/parity.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import { loadFiles, FetchPickPlace, Expert } from "../js/fetch_env.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const mujoco = await loadMujoco();
const xml = await loadFiles(mujoco, p => fs.readFileSync(path.join(root, p), "utf8"), p => new Uint8Array(fs.readFileSync(path.join(root, p))));
const env = new FetchPickPlace(mujoco, xml);
const ref = JSON.parse(fs.readFileSync(path.join(root, "assets/reference.json"), "utf8"));
const maxdiff = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
let fail = 0;
const check = (name, v, tol) => { const ok = v <= tol; if (!ok) fail++; console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${v.toExponential(2)} (tol ${tol})`); };

check("initial_qpos", maxdiff([...env.initialQpos], ref.initial_qpos), 1e-6);
check("initial_gripper_xpos", maxdiff(env.initialGripperXpos, ref.initial_gripper_xpos), 1e-6);
check("height_offset", Math.abs(env.heightOffset - ref.height_offset), 1e-6);

let worstGrip = 0, worstObj = 0, solvedReplay = 0, solvedExpert = 0;
for (const ep of ref.episodes) {
  env.reset({ cube: ep.cube, goal: ep.goal });
  let o;
  ep.actions.forEach((a, t) => {
    o = env.step(a);
    worstGrip = Math.max(worstGrip, maxdiff(o.grip, ep.grip[t]));
    worstObj = Math.max(worstObj, maxdiff(o.obj, ep.obj[t]));
  });
  solvedReplay += o.success ? 1 : 0;
  const e = new Expert(); o = env.reset({ cube: ep.cube, goal: ep.goal });
  for (let t = 0; t < 50; t++) o = env.step(e.act(o));
  solvedExpert += o.success ? 1 : 0;
}
check("replay: max gripper position error over 5x50 steps (m)", worstGrip, 1e-3);
check("replay: max cube position error over 5x50 steps (m)", worstObj, 5e-3);
console.log(`replayed Python actions: ${solvedReplay}/${ref.episodes.length} solved; JS expert closed-loop: ${solvedExpert}/${ref.episodes.length}`);
if (solvedExpert !== ref.episodes.length) fail++;
process.exit(fail ? 1 : 0);
