// three.js renderer that reproduces MuJoCo's legacy OpenGL look closely enough for the policy camera:
// per-geom rgba/specular/shininess from MuJoCo's own mjvScene (assets/render_ref.json), the headlight + light0,
// fixed-function Blinn-Phong, gradient skybox, 4x MSAA, same free camera. Layer 0 = what the robot sees,
// layer 1 = UI helpers only the visitor sees.
import * as THREE from "three";

THREE.ColorManagement.enabled = false;
export const IMG = 96;
const OBJ_GEOM = 5, OBJ_SITE = 6;

const VS = /* glsl */`
varying vec3 vN; varying vec3 vW;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const FS = /* glsl */`
uniform vec4 rgba; uniform float spec; uniform float shin; uniform float emis;
uniform vec3 hlDir; uniform vec3 hlAmb; uniform vec3 hlDif; uniform vec3 hlSpec;
uniform vec3 l0Dir; uniform vec3 l0Amb; uniform vec3 l0Dif; uniform vec3 l0Spec;
uniform vec3 globalAmb; uniform vec3 tint;
varying vec3 vN; varying vec3 vW;
vec3 light(vec3 n, vec3 v, vec3 dir, vec3 a, vec3 d, vec3 s) {
  vec3 L = -normalize(dir);
  float ndl = dot(n, L);
  vec3 c = a * rgba.rgb + d * rgba.rgb * max(ndl, 0.0);
  if (ndl > 0.0) c += s * spec * pow(max(dot(n, normalize(L + v)), 0.0), shin);
  return c;
}
void main() {
  vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
  vec3 v = normalize(cameraPosition - vW);
  vec3 c = globalAmb * rgba.rgb + emis * rgba.rgb
         + light(n, v, hlDir, hlAmb, hlDif, hlSpec) + light(n, v, l0Dir, l0Amb, l0Dif, l0Spec);
  gl_FragColor = vec4(clamp(c * tint, 0.0, 1.0), rgba.a);
}`;
const SKY_FS = /* glsl */`
uniform vec3 top; uniform vec3 bottom; varying vec3 vW;
void main() { vec3 d = normalize(vW - cameraPosition); gl_FragColor = vec4(mix(bottom, top, 0.5 + 0.5 * d.z), 1.0); }`;
const SKY_VS = /* glsl */`
varying vec3 vW;
void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; gl_Position.z = gl_Position.w * 0.99999; }`;

export class SimRenderer {
  /** env: FetchPickPlace, ref: render_ref.json, cam: PixelGrasp CAMERA dict. */
  constructor(env, ref, cam) {
    this.env = env; this.ref = ref;
    this.scene = new THREE.Scene();
    const shared = {
      hlDir: { value: new THREE.Vector3() }, hlAmb: { value: new THREE.Vector3(...ref.lights[0].ambient) },
      hlDif: { value: new THREE.Vector3(...ref.lights[0].diffuse) }, hlSpec: { value: new THREE.Vector3(...ref.lights[0].specular) },
      l0Dir: { value: new THREE.Vector3(...ref.lights[1].dir) }, l0Amb: { value: new THREE.Vector3(...ref.lights[1].ambient) },
      l0Dif: { value: new THREE.Vector3(...ref.lights[1].diffuse) }, l0Spec: { value: new THREE.Vector3(...ref.lights[1].specular) },
      globalAmb: { value: new THREE.Vector3(0, 0, 0) }, tint: { value: new THREE.Vector3(1, 1, 1) },
    };
    this.shared = shared;
    // sky
    const sky = new THREE.Mesh(new THREE.SphereGeometry(50, 32, 16), new THREE.ShaderMaterial({
      vertexShader: SKY_VS, fragmentShader: SKY_FS, side: THREE.BackSide, depthWrite: false,
      uniforms: { top: { value: new THREE.Vector3(...ref.skybox.rgb1) }, bottom: { value: new THREE.Vector3(...ref.skybox.rgb2) } } }));
    sky.frustumCulled = false; sky.renderOrder = -1; this.sky = sky; this.scene.add(sky);

    const m = env.model;
    this.items = [];
    for (const g of ref.geoms) {
      if (g.rgba[3] <= 0) continue;
      let geo;
      const s = g.size;
      if (g.type === 0) geo = new THREE.PlaneGeometry(2 * s[0], 2 * s[1], 10, 10);
      else if (g.type === 2) geo = new THREE.SphereGeometry(s[0], 24, 16);
      else if (g.type === 6) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      else if (g.type === 7) geo = this.meshGeometry(m, m.geom_dataid[g.objid]);
      else continue;
      const mat = new THREE.ShaderMaterial({
        vertexShader: VS, fragmentShader: FS, transparent: g.rgba[3] < 1, depthWrite: g.rgba[3] >= 1,
        uniforms: { ...shared, rgba: { value: new THREE.Vector4(...g.rgba) }, spec: { value: g.specular },
          shin: { value: g.shininess * 128 }, emis: { value: g.emission } } });
      const mesh = new THREE.Mesh(geo, mat); mesh.matrixAutoUpdate = false; mesh.frustumCulled = false;
      this.scene.add(mesh);
      const it = { mesh, objtype: g.objtype, objid: g.objid, name: g.name, rgba: mat.uniforms.rgba.value };
      this.items.push(it);
      if (g.objtype === OBJ_SITE && g.name === "target0") this.target = it;
      if (g.objtype === OBJ_GEOM && g.name === "object0") this.cube = it;
    }
    // policy camera (identical to the training camera: free camera, fovy 45, square)
    const az = cam.azimuth * Math.PI / 180, el = cam.elevation * Math.PI / 180;
    const fwd = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
    const look = new THREE.Vector3(...cam.lookat);
    this.policyCam = new THREE.PerspectiveCamera(ref.fovy, 1, ref.camera.frustum_near, ref.camera.frustum_far);
    this.policyCam.up.set(0, 0, 1);
    this.policyCam.position.copy(look.clone().addScaledVector(fwd, -cam.distance));
    this.policyCam.lookAt(look);
    this.policyCam.layers.set(0);
    this.policyFwd = fwd;
    this.rt = new THREE.WebGLRenderTarget(IMG, IMG, { samples: 4, type: THREE.UnsignedByteType });
    this.pix = new Uint8Array(IMG * IMG * 4);
    this.img = new Uint8Array(IMG * IMG * 3);
  }

  meshGeometry(m, meshId) {
    const va = m.mesh_vertadr[meshId], fa = m.mesh_faceadr[meshId], fn = m.mesh_facenum[meshId], na = m.mesh_normaladr[meshId];
    const V = m.mesh_vert, F = m.mesh_face, N = m.mesh_normal, FN = m.mesh_facenormal;
    const pos = new Float32Array(fn * 9), nor = new Float32Array(fn * 9);
    for (let f = 0; f < fn; f++) for (let k = 0; k < 3; k++) {
      const vi = va + F[3 * (fa + f) + k], ni = na + FN[3 * (fa + f) + k];
      for (let c = 0; c < 3; c++) { pos[9 * f + 3 * k + c] = V[3 * vi + c]; nor[9 * f + 3 * k + c] = N[3 * ni + c]; }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    return geo;
  }

  /** Copy MuJoCo poses into the three.js scene. The red target marker is drawn at env.goal. */
  sync() {
    const d = this.env.data;
    const gx = d.geom_xpos, gm = d.geom_xmat, sx = d.site_xpos, sm = d.site_xmat;
    for (const it of this.items) {
      const P = it.objtype === OBJ_GEOM ? gx : sx, R = it.objtype === OBJ_GEOM ? gm : sm, i = it.objid;
      let px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
      if (it === this.target) [px, py, pz] = this.env.goal;
      it.mesh.matrix.set(R[9 * i], R[9 * i + 1], R[9 * i + 2], px, R[9 * i + 3], R[9 * i + 4], R[9 * i + 5], py,
        R[9 * i + 6], R[9 * i + 7], R[9 * i + 8], pz, 0, 0, 0, 1);
      it.mesh.matrixWorldNeedsUpdate = true;
    }
  }

  setHeadlight(camera) {
    camera.getWorldDirection(this.shared.hlDir.value);
  }

  /** Render the 96x96 robot camera image; returns HWC uint8 RGB, top row first (same as env.render()). */
  renderPolicy(renderer) {
    this.sync();
    this.shared.hlDir.value.copy(this.policyFwd);
    renderer.setRenderTarget(this.rt);
    renderer.render(this.scene, this.policyCam);
    renderer.readRenderTargetPixels(this.rt, 0, 0, IMG, IMG, this.pix);
    renderer.setRenderTarget(null);
    for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
      const s = 4 * ((IMG - 1 - y) * IMG + x), t = 3 * (y * IMG + x);
      this.img[t] = this.pix[s]; this.img[t + 1] = this.pix[s + 1]; this.img[t + 2] = this.pix[s + 2];
    }
    return this.img;
  }

  /** Render the visitor's view (all layers). */
  renderView(renderer, camera) {
    this.sync();
    this.setHeadlight(camera);
    renderer.render(this.scene, camera);
  }
}
