// PixelGrasp live demo: MuJoCo physics (WASM) + the trained camera-only policy (ONNX in a Web Worker).
// The visitor drags the cube and the red target; the arm returns home, looks through its camera and picks it up.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import loadMujoco from "../vendor/mujoco/mujoco.js";
import { loadFiles, FetchPickPlace, Expert, DRAG } from "./fetch_env.js";
import { SimRenderer, IMG } from "./render.js";

const $ = id => document.getElementById(id);
const STEP_MS = 40;            // one env step = 20 substeps x 2 ms of simulated time
const EPISODE = 50;            // steps per attempt, as in training/eval
const HOME_STEPS = 30;
const status = (t, cls = "") => { const s = $("status"); s.textContent = t; s.className = cls; };

async function main() {
  status("Loading physics…");
  const mujoco = await loadMujoco();
  const txt = async p => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.text(); };
  const bin = async p => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
  const env = new FetchPickPlace(mujoco, await loadFiles(mujoco, txt, bin));
  const [cam, rref] = await Promise.all([fetch("assets/reference.json").then(r => r.json()).then(r => r.camera), fetch("assets/render_ref.json").then(r => r.json())]);

  // ---------- rendering ----------
  const canvas = $("view");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const R = new SimRenderer(env, rref, cam);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 60);
  camera.up.set(0, 0, 1);
  camera.position.set(2.45, 0.1, 1.25);
  camera.layers.enableAll();
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(1.28, 0.78, 0.47);
  controls.enableDamping = true; controls.minDistance = 0.6; controls.maxDistance = 3.5; controls.maxPolarAngle = Math.PI * 0.49;
  controls.update();
  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas); resize();

  // visitor-only helpers (layer 1: never in the robot's camera image)
  const helper = (obj) => { obj.layers.set(1); obj.traverse(o => o.layers.set(1)); R.scene.add(obj); return obj; };
  const ringMat = c => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.85, depthTest: false });
  const cubeRing = helper(new THREE.Mesh(new THREE.RingGeometry(0.034, 0.042, 40), ringMat(0x4cc9f0)));
  const goalRing = helper(new THREE.Mesh(new THREE.RingGeometry(0.03, 0.036, 40), ringMat(0xff4d4d)));
  const goalStem = helper(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({ color: 0xff4d4d, dashSize: 0.01, gapSize: 0.008, depthTest: false, transparent: true })));
  cubeRing.renderOrder = goalRing.renderOrder = goalStem.renderOrder = 10;
  const g0 = env.initialGripperXpos, zt = env.heightOffset - 0.0249;
  const [[x0, x1], [y0, y1]] = [DRAG.x, DRAG.y];
  const cubeZone = helper(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([dx, dy]) => new THREE.Vector3(g0[0] + dx, g0[1] + dy, zt + 0.0008))),
    new THREE.LineBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 0.7, depthTest: false })));
  cubeZone.visible = false;
  const pickMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
  const cubePick = helper(new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 8), pickMat));
  const goalPick = helper(new THREE.Mesh(new THREE.SphereGeometry(0.04, 12, 8), pickMat));

  const eye = $("eye"), eyeCtx = eye.getContext("2d"), eyeImg = eyeCtx.createImageData(IMG, IMG);
  const showEye = img => {
    for (let i = 0, j = 0; i < IMG * IMG * 3; i += 3, j += 4) { eyeImg.data[j] = img[i]; eyeImg.data[j + 1] = img[i + 1]; eyeImg.data[j + 2] = img[i + 2]; eyeImg.data[j + 3] = 255; }
    eyeCtx.putImageData(eyeImg, 0, 0);
  };

  // ---------- policy worker ----------
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  let policyReady = false, pending = false, buf = [], runId = 0, lastMs = 0, msHist = [];
  worker.onmessage = ({ data: m }) => {
    if (m.type === "progress") status(`Downloading the trained policy… ${Math.round(m.frac * 100)}% of 46 MB`);
    if (m.type === "ready") { policyReady = true; $("model").textContent = `ResNet-18 · ${(m.meta.bytes / 1e6).toFixed(0)} MB ONNX · chunk ${m.meta.chunk}, re-plans every ${m.meta.exec_steps}`; startFresh(); }
    if (m.type === "chunk" && m.id === runId) { buf = m.actions.slice(); pending = false; lastMs = m.ms; msHist.push(m.ms); if (msHist.length > 30) msHist.shift(); }
    if (m.type === "error") { status("Policy error: " + m.message, "bad"); console.error(m.message); }
  };
  worker.postMessage({ type: "load", url: new URL("models/pixelgrasp_k8.onnx", location.href).href, metaUrl: new URL("models/pixelgrasp_k8.json", location.href).href });

  // ---------- episode logic ----------
  let mode = "loading";        // loading | homing | running | holding | idle
  let brain = "policy";        // policy | expert
  let obs = env.reset(env.randomLayout()), steps = 0, homeN = 0, expert = new Expert();
  let tries = 0, wins = 0, dragging = null, everSucceeded = false, holdAt = null;
  const cubeLifted = () => obs.obj[2] > env.heightOffset + 0.015;

  function startFresh(layout) {
    obs = env.reset(layout || env.randomLayout());
    beginRun();
  }
  function beginHoming() { runId++; pending = false; buf = []; mode = "homing"; homeN = 0; status("Going back to the start pose…"); }
  function beginRun() {
    runId++; pending = false; buf = []; expert.reset(); steps = 0; everSucceeded = false;
    mode = "running"; tries++; updateScore();
    status(brain === "policy" ? "Looking at the camera image and going for the cube…" : "Scripted expert (cheats: reads the cube position)…");
  }
  function finishRun() {
    const ok = obs.success;
    if (!ok && obs.obj[2] < env.heightOffset - 0.05) {     // knocked off the table: put it back so the next try is possible
      const [x, y] = env.clampCube(env.initialGripperXpos[0] + 0.12, env.initialGripperXpos[1]); env.placeCube(x, y); obs = env.obs();
      mode = "idle"; updateScore(); status("It knocked the cube off the table, so I put it back. Press Retry or drag it.", "bad"); return;
    }
    if (ok) { wins++; mode = "holding"; holdAt = [...obs.grip]; status("Done. Drag the cube, or click the table to move the target.", "good"); }
    else { mode = "idle"; status(`Missed (cube ${(obs.dist * 100).toFixed(0)} cm from the target). Drag something or press Retry.`, "bad"); }
    updateScore();
  }
  function updateScore() {
    $("score").textContent = tries ? `${wins} / ${tries}` : "–";
    $("brain-note").textContent = brain === "policy" ? "Trained policy: sees only the camera image + its own gripper position."
      : "Scripted expert: reads the exact cube position from the simulator (what the policy learned from).";
  }

  function tick() {
    if (dragging === "cube" || mode === "loading" || mode === "idle") return false;
    if (mode === "homing") {
      const gp = obs.grip, h = env.initialGripperXpos;
      const a = [0, 1, 2].map(k => Math.max(-1, Math.min(1, 10 * (h[k] - gp[k])))).concat([1]);
      obs = env.step(a); homeN++;
      if (Math.hypot(h[0] - gp[0], h[1] - gp[1], h[2] - gp[2]) < 0.012 || homeN >= HOME_STEPS) { env.homeRobot(); obs = env.obs(); beginRun(); }
      return true;
    }
    if (mode === "holding") {       // keep the cube steady at the goal: P-control to the pose where it succeeded
      const gp = obs.grip;
      obs = env.step([0, 1, 2].map(k => Math.max(-1, Math.min(1, 10 * (holdAt[k] - gp[k])))).concat([obs.success || cubeLifted() ? -1 : 1]));
      return true;
    }
    // running
    let a;
    if (brain === "expert") {
      a = expert.act(obs);
      if (steps % 4 === 0) showEye(R.renderPolicy(renderer));
    } else {
      if (!buf.length) {
        if (!pending) {
          pending = true;
          const img = R.renderPolicy(renderer).slice(); showEye(img);
          worker.postMessage({ type: "act", id: runId, image: img, state: env.stateVec(obs) }, [img.buffer]);
        }
        return false;
      }
      a = buf.shift();
    }
    obs = env.step(a); steps++;
    everSucceeded ||= obs.success;
    $("steps").textContent = `${steps} / ${EPISODE}`;
    if (steps >= EPISODE) finishRun();
    return true;
  }

  // ---------- dragging ----------
  const ray = new THREE.Raycaster(); ray.layers.enableAll();
  const ndc = new THREE.Vector2(), hit = new THREE.Vector3();
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const setNdc = e => { const r = canvas.getBoundingClientRect(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); };
  let downAt = null;
  canvas.addEventListener("pointerdown", e => {
    downAt = [e.clientX, e.clientY];
    if (mode === "loading") return;
    setNdc(e); ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects([cubePick, goalPick], false);
    if (!hits.length) return;
    // when the cube sits on the target both are hit: the cube wins (move the target by clicking the table)
    dragging = hits.some(h => h.object === cubePick) ? "cube" : "target";
    controls.enabled = false; canvas.setPointerCapture(e.pointerId); canvas.classList.add("grabbing");
    plane.constant = -(dragging === "cube" ? env.heightOffset : env.goal[2]);
    if (dragging === "cube") { runId++; pending = false; buf = []; cubeZone.visible = true; status("Put the cube anywhere inside the blue box."); }
  });
  canvas.addEventListener("pointermove", e => {
    setNdc(e);
    if (!dragging) {
      ray.setFromCamera(ndc, camera);
      canvas.classList.toggle("can-grab", !!ray.intersectObjects([cubePick, goalPick], false).length);
      return;
    }
    ray.setFromCamera(ndc, camera);
    if (!ray.ray.intersectPlane(plane, hit)) return;
    if (dragging === "cube") { const [x, y] = env.clampCube(hit.x, hit.y); env.placeCube(x, y); obs = env.obs(); }
    else { env.setGoal(env.clampGoal(hit.x, hit.y, env.goal[2])); obs = env.obs(); }
  });
  const endDrag = e => {
    const click = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5; downAt = null;
    if (!dragging && click && e.type === "pointerup" && mode !== "loading") {     // click on the table = put the target there
      setNdc(e); ray.setFromCamera(ndc, camera); plane.constant = -env.heightOffset;
      const g = env.initialGripperXpos;
      if (ray.ray.intersectPlane(plane, hit) && Math.abs(hit.x - g[0]) < 0.3 && Math.abs(hit.y - g[1]) < 0.35) {
        env.setGoal(env.clampGoal(hit.x, hit.y, env.goal[2])); obs = env.obs(); onGoalMoved();
      }
      return;
    }
    if (!dragging) return;
    const was = dragging; dragging = null; controls.enabled = true; cubeZone.visible = false; canvas.classList.remove("grabbing");
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    if (was === "cube") beginHoming();
    else onGoalMoved();
  };
  canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag);

  function onGoalMoved() {
    $("height").value = Math.round((env.goal[2] - env.heightOffset) * 100);
    $("height-val").textContent = `${$("height").value} cm`;
    if (mode === "running") { steps = Math.min(steps, 10); return; }     // give it time to reach the new spot
    if (mode === "homing") return;
    if (cubeLifted()) { runId++; buf = []; pending = false; expert.reset(); expert.phase = 3; steps = 0; mode = "running"; tries++; updateScore(); status("Carrying it to the new spot…"); }
    else beginHoming();
  }
  $("height").addEventListener("input", () => {
    $("height-val").textContent = `${$("height").value} cm`;
    env.setGoal(env.clampGoal(env.goal[0], env.goal[1], env.heightOffset + $("height").value / 100)); obs = env.obs();
  });
  $("height").addEventListener("change", onGoalMoved);
  $("shuffle").onclick = () => { if (mode !== "loading") startFresh(); };
  $("retry").onclick = () => { if (mode !== "loading") beginHoming(); };
  for (const b of document.querySelectorAll("[data-brain]")) b.onclick = () => {
    brain = b.dataset.brain;
    document.querySelectorAll("[data-brain]").forEach(x => x.setAttribute("aria-pressed", x === b));
    updateScore(); if (mode !== "loading") beginHoming();
  };

  // ---------- main loop ----------
  let acc = 0, last = performance.now();
  status("Downloading the trained policy…");
  function frame(now) {
    acc += Math.min(120, now - last); last = now;
    while (acc >= STEP_MS) { if (!tick()) { acc = 0; break; } acc -= STEP_MS; }
    // helpers follow the sim
    const o = env.obs();
    cubeRing.position.set(o.obj[0], o.obj[1], zt + 0.001); cubePick.position.set(...o.obj);
    goalPick.position.set(...env.goal); goalRing.position.set(env.goal[0], env.goal[1], zt + 0.001);
    goalStem.geometry.setFromPoints([new THREE.Vector3(env.goal[0], env.goal[1], zt), new THREE.Vector3(...env.goal)]); goalStem.computeLineDistances();
    goalStem.visible = env.goal[2] > env.heightOffset + 0.01;
    $("dist").textContent = `${(o.dist * 100).toFixed(1)} cm`;
    $("ms").textContent = brain === "policy" && msHist.length ? `${Math.round(msHist.reduce((a, b) => a + b, 0) / msHist.length)} ms` : "–";
    if (mode !== "running") $("steps").textContent = mode === "homing" ? "homing" : "–";
    controls.update();
    R.sky.visible = false; R.scene.background = BG;
    R.renderView(renderer, camera);
    R.sky.visible = true; R.scene.background = null;
    requestAnimationFrame(frame);
  }
  const BG = new THREE.Color(0x0e1116);
  updateScore();
  requestAnimationFrame(frame);
  window.pg = { env, R, camera, get mode() { return mode; }, get tries() { return tries; }, get wins() { return wins; }, startFresh, beginHoming,
    /** screen (client) coords of a world point, for tests */
    project(x, y, z) { const r = canvas.getBoundingClientRect(), v = new THREE.Vector3(x, y, z).project(camera); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; } };
}

main().catch(e => { status("Could not start: " + e.message, "bad"); console.error(e); });
