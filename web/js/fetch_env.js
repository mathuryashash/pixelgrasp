// FetchPickAndPlace-v4 (gymnasium-robotics 1.4.2) ported to MuJoCo WASM.
// Mirrors fetch_env.py / robot_env.py: _env_setup, _reset_sim, _set_action (mocap + position actuators), 20 substeps,
// observation (grip site, object site, finger qpos) and the 5 cm success test. Works in the browser and in Node.
export const N_SUBSTEPS = 20;
export const DIST_THRESHOLD = 0.05;
export const OBJ_RANGE = 0.15, TARGET_RANGE = 0.15;
// Where the visitor may drop the cube (offsets from the start gripper position). Training used +-15 cm.
// Measured with the trained policy from the home pose against a hard target grid (9 xy x 2 heights per spot, see
// web/test/range): +-17 cm holds 17-18/18 on every edge and corner; 20 cm drops to 12-16/18; 25 cm fails outright.
// +x (away from the robot) stays at the training edge: the table ends at +21 cm and 16 cm already drops to 14-17/18.
export const DRAG = { x: [-0.17, 0.15], y: [-0.17, 0.17] };
const INITIAL_QPOS = { "robot0:slide0": [0.405], "robot0:slide1": [0.48], "robot0:slide2": [0.0],
  "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0] };
const OBJ = { BODY: 1, JOINT: 3, GEOM: 5, SITE: 6, EQUALITY: 17, ACTUATOR: 19 };

export async function loadFiles(mujoco, fetchText, fetchBytes, base = "assets") {
  const manifest = JSON.parse(await fetchText(`${base}/manifest.json`));
  const FS = mujoco.FS;
  for (const d of ["/working", "/working/meshes", "/working/textures"]) { try { FS.mkdir(d); } catch (e) { /* exists */ } }
  FS.writeFile("/working/pick_and_place.xml", await fetchText(`${base}/pick_and_place.xml`));
  await Promise.all(manifest.meshes.map(async m => FS.writeFile(`/working/meshes/${m}`, await fetchBytes(`${base}/meshes/${m}`))));
  await Promise.all(manifest.textures.map(async t => FS.writeFile(`/working/textures/${t}`, await fetchBytes(`${base}/textures/${t}`))));
  return "/working/pick_and_place.xml";
}

export class FetchPickPlace {
  constructor(mujoco, xmlPath) {
    this.mj = mujoco;
    this.model = mujoco.MjModel.from_xml_path(xmlPath);
    this.data = new mujoco.MjData(this.model);
    const id = (t, n) => { const i = mujoco.mj_name2id(this.model, t, n); if (i < 0) throw new Error(`no ${n}`); return i; };
    this.id = id;
    this.gripSite = id(OBJ.SITE, "robot0:grip");
    this.objSite = id(OBJ.SITE, "object0");
    this.targetSite = id(OBJ.SITE, "target0");
    this.gripperBody = id(OBJ.BODY, "robot0:gripper_link");
    this.objQ = this.model.jnt_qposadr[id(OBJ.JOINT, "object0:joint")];
    this.objV = this.model.jnt_dofadr[id(OBJ.JOINT, "object0:joint")];
    // robot_get_obs: last two joints whose name starts with "robot" (model order: r finger, l finger)
    this.fingerQ = ["robot0:r_gripper_finger_joint", "robot0:l_gripper_finger_joint"].map(n => this.model.jnt_qposadr[id(OBJ.JOINT, n)]);
    this.goal = [0, 0, 0];
    this._envSetup();
  }

  qposOfJoint(name) { return this.model.jnt_qposadr[this.id(OBJ.JOINT, name)]; }

  _envSetup() {
    const { mj, model, data } = this;
    for (const [n, v] of Object.entries(INITIAL_QPOS)) { const a = this.qposOfJoint(n); v.forEach((x, k) => { data.qpos[a + k] = x; }); }
    // reset_mocap_welds: eq_data[i, :7] = [0,0,0, 0,0,0,1] for weld constraints
    const NEQ = model.eq_data.length / model.neq;
    for (let i = 0; i < model.neq; i++) {
      if (model.eq_type[i] === 1 /* mjEQ_WELD */) [0, 0, 0, 0, 0, 0, 1].forEach((x, k) => { model.eq_data[i * NEQ + k] = x; });
    }
    mj.mj_forward(model, data);
    const g = this.site(this.gripSite);
    this.setMocap([-0.498 + g[0], 0.005 + g[1], -0.431 + 0.2 + g[2]], [1, 0, 1, 0]);
    for (let i = 0; i < 10; i++) this.substeps();
    this.initialGripperXpos = this.site(this.gripSite);
    this.heightOffset = this.site(this.objSite)[2];
    this.initialTime = data.time;
    this.initialQpos = Float64Array.from(data.qpos);
    this.initialQvel = Float64Array.from(data.qvel);
  }

  site(i) { const p = this.data.site_xpos; return [p[3 * i], p[3 * i + 1], p[3 * i + 2]]; }
  setMocap(pos, quat) { const d = this.data; pos.forEach((x, k) => { d.mocap_pos[k] = x; }); quat.forEach((x, k) => { d.mocap_quat[k] = x; }); }
  substeps() { for (let i = 0; i < N_SUBSTEPS; i++) this.mj.mj_step(this.model, this.data); }

  /** layout = {cube:[x,y,(z)], goal:[x,y,z]}. Same as _reset_sim with the cube xy given instead of sampled. */
  reset(layout) {
    const { mj, model, data } = this;
    mj.mj_resetData(model, data);
    data.time = this.initialTime;
    data.qpos.set(this.initialQpos);
    data.qvel.set(this.initialQvel);
    if (layout) { data.qpos[this.objQ] = layout.cube[0]; data.qpos[this.objQ + 1] = layout.cube[1]; this.goal = [...layout.goal]; }
    mj.mj_forward(model, data);
    return this.obs();
  }

  /** Move the cube while the episode runs (the visitor dragging it). Puts it flat on the table at rest. */
  placeCube(x, y) {
    const { data } = this;
    const q = [x, y, this.heightOffset, 1, 0, 0, 0];
    q.forEach((v, k) => { data.qpos[this.objQ + k] = v; });
    for (let k = 0; k < 6; k++) data.qvel[this.objV + k] = 0;
    this.mj.mj_forward(this.model, data);
  }

  setGoal(g) { this.goal = [...g]; }

  /** Put the robot back in its exact start pose (same as reset) but leave the cube where it is. */
  homeRobot() {
    const { data } = this;
    const cq = Array.from(data.qpos.subarray(this.objQ, this.objQ + 7)), cv = Array.from(data.qvel.subarray(this.objV, this.objV + 6));
    data.qpos.set(this.initialQpos); data.qvel.set(this.initialQvel);
    cq.forEach((v, k) => { data.qpos[this.objQ + k] = v; }); cv.forEach((v, k) => { data.qvel[this.objV + k] = v; });
    this.mj.mj_forward(this.model, data);
  }

  /** Clamp a dragged cube to the drag area (DRAG offsets around the start gripper), >= 10 cm from it. */
  clampCube(x, y, box = DRAG) {
    const g = this.initialGripperXpos;
    let dx = Math.min(box.x[1], Math.max(box.x[0], x - g[0])), dy = Math.min(box.y[1], Math.max(box.y[0], y - g[1]));
    const r = Math.hypot(dx, dy);
    if (r < 0.1) { const s = r < 1e-6 ? 0 : 0.1 / r; dx = r < 1e-6 ? 0.1 : dx * s; dy = r < 1e-6 ? 0 : dy * s; }
    return [g[0] + dx, g[1] + dy];
  }

  /** Clamp a goal to the training goal box: xy within +-15 cm, z from table height to 45 cm above. */
  clampGoal(x, y, z) {
    const g = this.initialGripperXpos;
    return [g[0] + Math.min(TARGET_RANGE, Math.max(-TARGET_RANGE, x - g[0])), g[1] + Math.min(TARGET_RANGE, Math.max(-TARGET_RANGE, y - g[1])),
      Math.min(this.heightOffset + 0.45, Math.max(this.heightOffset, z))];
  }

  /** Random layout drawn the way the env samples it (cube >= 10 cm from gripper, goal half the time in the air). */
  randomLayout(rand = Math.random) {
    const g = this.initialGripperXpos, u = () => (rand() * 2 - 1);
    let c; do { c = [g[0] + u() * OBJ_RANGE, g[1] + u() * OBJ_RANGE]; } while (Math.hypot(c[0] - g[0], c[1] - g[1]) < 0.1);
    const goal = [g[0] + u() * TARGET_RANGE, g[1] + u() * TARGET_RANGE, this.heightOffset + (rand() < 0.5 ? rand() * 0.45 : 0)];
    return { cube: c, goal };
  }

  step(action) {
    const { model, data } = this;
    const a = action.map(x => Math.min(1, Math.max(-1, x)));
    // ctrl_set_action: position actuators -> ctrl = qpos[joint] + gripper action
    for (let i = 0; i < model.nu; i++) {
      const j = model.actuator_trnid[2 * i];
      data.ctrl[i] = model.actuator_biastype[i] === 0 ? a[3] : data.qpos[model.jnt_qposadr[j]] + a[3];
    }
    // mocap_set_action: mocap = welded body pose + delta (pos * 0.05, quat + [1,0,1,0])
    const b = this.gripperBody, xp = data.xpos, xq = data.xquat;
    this.setMocap([xp[3 * b] + a[0] * 0.05, xp[3 * b + 1] + a[1] * 0.05, xp[3 * b + 2] + a[2] * 0.05],
      [xq[4 * b] + 1, xq[4 * b + 1], xq[4 * b + 2] + 1, xq[4 * b + 3]]);
    this.substeps();
    return this.obs();
  }

  obs() {
    const grip = this.site(this.gripSite), obj = this.site(this.objSite);
    const fingers = this.fingerQ.map(a => this.data.qpos[a]);
    const d = Math.hypot(obj[0] - this.goal[0], obj[1] - this.goal[1], obj[2] - this.goal[2]);
    return { grip, obj, fingers, goal: [...this.goal], dist: d, success: d < DIST_THRESHOLD };
  }

  /** Policy state vector: gripper xyz, 2 finger widths, goal xyz (PixelGrasp policy_vec with use_goal). */
  stateVec(o = this.obs()) { return Float32Array.from([...o.grip, ...o.fingers, ...o.goal]); }

  /** Same ranges the env samples from (used to clamp where the visitor may drop the cube / target). */
  bounds() {
    const g = this.initialGripperXpos;
    return { x: [g[0] - OBJ_RANGE, g[0] + OBJ_RANGE], y: [g[1] - OBJ_RANGE, g[1] + OBJ_RANGE], table: this.heightOffset, zmax: this.heightOffset + 0.45 };
  }
}

/** The scripted expert from pixelgrasp/env.py (privileged state), used for the parity test and as a comparison mode. */
export class Expert {
  constructor(gain = 10) { this.gain = gain; this.phase = 0; }
  reset() { this.phase = 0; }
  act(o) {
    const { grip, obj, goal, fingers } = o; let target, g;
    const n = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    if (this.phase === 0) { target = [obj[0], obj[1], obj[2] + 0.06]; g = 1; if (n(target, grip) < 0.01) this.phase = 1; }
    if (this.phase === 1) { target = [obj[0], obj[1], obj[2] - 0.005]; g = 1; if (n(target, grip) < 0.008) this.phase = 2; }
    if (this.phase === 2) { target = grip; g = -1; if (fingers[0] + fingers[1] < 0.052) this.phase = 3; }
    if (this.phase === 3) { target = goal; g = -1; }
    return [0, 1, 2].map(k => Math.min(1, Math.max(-1, this.gain * (target[k] - grip[k])))).concat([g]);
  }
}
