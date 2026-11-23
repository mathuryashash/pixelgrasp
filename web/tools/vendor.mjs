// Copy the runtime files the page needs out of node_modules into web/vendor (no bundler).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const nm = path.join(root, "node_modules"), out = path.join(root, "vendor");
const files = {
  "mujoco/mujoco.js": "@mujoco/mujoco/mujoco.js",
  "mujoco/mujoco.wasm": "@mujoco/mujoco/mujoco.wasm",
  "three/three.module.js": "three/build/three.module.js",
  "three/three.core.js": "three/build/three.core.js",
  "three/addons/controls/OrbitControls.js": "three/examples/jsm/controls/OrbitControls.js",
  "ort/ort.wasm.min.mjs": "onnxruntime-web/dist/ort.wasm.min.mjs",
  "ort/ort-wasm-simd-threaded.mjs": "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
  "ort/ort-wasm-simd-threaded.wasm": "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
};
for (const [dst, src] of Object.entries(files)) {
  fs.mkdirSync(path.dirname(path.join(out, dst)), { recursive: true });
  fs.copyFileSync(path.join(nm, src), path.join(out, dst));
}
console.log(`vendored ${Object.keys(files).length} files into ${out}`);
