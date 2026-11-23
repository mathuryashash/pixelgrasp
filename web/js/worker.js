// Policy inference off the main thread so dragging and rendering stay smooth.
// Messages in:  {type:"load", url, metaUrl} | {type:"act", id, image:Uint8Array(96*96*3), state:Float32Array(8)}
// Messages out: {type:"progress", frac} | {type:"ready", meta} | {type:"chunk", id, actions, ms} | {type:"error", message}
import * as ort from "../vendor/ort/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
let session = null, meta = null;

async function load(url, metaUrl) {
  meta = await (await fetch(metaUrl)).json();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || meta.bytes;
  const reader = res.body.getReader(); const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    parts.push(value); got += value.length; postMessage({ type: "progress", frac: Math.min(1, got / total) });
  }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  // warm-up so the first real decision is not slow
  await session.run({ image: new ort.Tensor("uint8", new Uint8Array(96 * 96 * 3), [1, 96, 96, 3]),
    state: new ort.Tensor("float32", new Float32Array(8), [1, 8]) });
  postMessage({ type: "ready", meta });
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === "load") await load(m.url, m.metaUrl);
    if (m.type === "act") {
      const t0 = performance.now();
      const out = await session.run({ image: new ort.Tensor("uint8", m.image, [1, 96, 96, 3]),
        state: new ort.Tensor("float32", m.state, [1, 8]) });
      const pos = out.pos.data, g = out.grip_logit.data, actions = [];
      for (let k = 0; k < meta.exec_steps; k++) actions.push([pos[3 * k], pos[3 * k + 1], pos[3 * k + 2], g[k] > 0 ? 1 : -1]);
      postMessage({ type: "chunk", id: m.id, actions, ms: performance.now() - t0 });
    }
  } catch (e) { postMessage({ type: "error", message: String(e && e.message || e) }); }
};
