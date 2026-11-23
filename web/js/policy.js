// The trained PixelGrasp policy in the browser (onnxruntime-web, WASM backend).
// Same closed loop as pixelgrasp/policy.py BCPolicyRunner: predict an 8-step action chunk from (96x96 image, state),
// execute the first `exec_steps` (4) of it, then re-plan.
import * as ort from "../vendor/ort/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export class Policy {
  static async load(url, metaUrl, onProgress) {
    const meta = await (await fetch(metaUrl)).json();
    const res = await fetch(url);
    const total = +res.headers.get("content-length") || meta.bytes;
    const reader = res.body.getReader(); const parts = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      parts.push(value); got += value.length; onProgress?.(got / total);
    }
    const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
    const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return new Policy(session, meta);
  }

  constructor(session, meta) { this.session = session; this.meta = meta; this.buf = []; this.lastMs = 0; }
  reset() { this.buf = []; }

  async predict(imgHWC, state) {
    const t0 = performance.now();
    const out = await this.session.run({
      image: new ort.Tensor("uint8", imgHWC, [1, 96, 96, 3]),
      state: new ort.Tensor("float32", state, [1, 8]),
    });
    this.lastMs = performance.now() - t0;
    const pos = out.pos.data, g = out.grip_logit.data, K = this.meta.chunk, acts = [];
    for (let k = 0; k < K; k++) acts.push([pos[3 * k], pos[3 * k + 1], pos[3 * k + 2], g[k] > 0 ? 1 : -1]);
    return acts;
  }

  /** One action for this control step; renders + infers only when the chunk buffer is empty. */
  async act(getImage, state) {
    let replanned = false;
    if (!this.buf.length) { this.buf = (await this.predict(getImage(), state)).slice(0, this.meta.exec_steps); replanned = true; }
    return { action: this.buf.shift(), replanned };
  }
}
