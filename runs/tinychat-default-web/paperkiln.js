/* paperkiln.js: a paperkiln word-level chat model, run in plain JavaScript.
 *
 * No WASM, no build step, no dependencies. The same file runs in a browser
 * (window.Paperkiln) and in Node (require("./paperkiln.js")), so the parity
 * test drives exactly the code visitors run.
 *
 * What it mirrors, op for op, in float32 storage:
 *   - tokenizer      include/microtorch/word_tokenizer.hpp (byte-wise, ASCII
 *                    lower-casing, "<|endoftext|>" as one token, unknown -> 0)
 *   - llama family   src/llama.cpp: RMSNorm (eps 1e-5, as ops::rmsnorm uses),
 *                    RoPE on adjacent pairs (every head, or head 0 only
 *                    for runs recorded as rope_heads "first"), SwiGLU, tied
 *                    head
 *   - flex family    tools/parity_model.hpp FlexLM (gpt2-small and the
 *                    paper-faithful decoder): LayerNorm (eps 1e-5) or RMSNorm,
 *                    GELU (tanh form) / ReLU / SwiGLU, learned or sinusoidal
 *                    positions, residual / highway / plain sublayers
 *   - gpt2 family    ParityLM(EXACT): the 2-block form of the same decoder
 *   - sampler        tools/mtstudio.cpp LoadedLM::next (special tokens banned,
 *                    temperature, top-k, top-p; greedy = first maximum)
 *   - chat           mtstudio chat: "user: <text> assistant:", history in the
 *                    same form, stop at end-of-text or a new turn marker
 *
 * Matrices are row-major Float32Array. Linear weights are [in, out] (the
 * engine computes x @ W), embeddings [vocab, d]. Context beyond T slides
 * exactly as the engine does: the last T ids, positions from 0. A KV cache
 * serves every step whose window extends the cached one; when the window
 * slides, the window is recomputed (as the engine recomputes it).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Paperkiln = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const f32 = Math.fround;

  // ---- safetensors ---------------------------------------------------------
  // 8-byte little-endian header length, a JSON header, then raw little-endian
  // tensor bytes. F32 is read directly; F16 and BF16 are widened to float32.
  function f16ToF32(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * Math.pow(2, -24);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }

  function parseSafetensors(buf) {
    const ab = buf instanceof ArrayBuffer ? buf : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    if (ab.byteLength < 8) throw new Error("safetensors: file too short");
    const dv = new DataView(ab);
    const lo = dv.getUint32(0, true), hi = dv.getUint32(4, true);
    if (hi !== 0 || lo === 0 || 8 + lo > ab.byteLength) throw new Error("safetensors: implausible header length");
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 8, lo)));
    const base = 8 + lo;
    const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
    const out = {};
    for (const [rawName, meta] of Object.entries(header)) {
      if (rawName === "__metadata__") continue;
      const name = rawName.startsWith("transformer.") ? rawName.slice(12) : rawName;
      const shape = meta.shape.map(Number);
      const [o0, o1] = meta.data_offsets.map(Number);
      const n = shape.reduce((a, b) => a * b, 1);
      const start = base + o0, nbytes = o1 - o0;
      if (start + nbytes > ab.byteLength) throw new Error("safetensors: " + name + " runs past the end of the file");
      let data;
      if (meta.dtype === "F32") {
        if (nbytes !== 4 * n) throw new Error("safetensors: size mismatch at " + name);
        if (littleEndian && start % 4 === 0) data = new Float32Array(ab, start, n);
        else {
          data = new Float32Array(n);
          for (let i = 0; i < n; i++) data[i] = dv.getFloat32(start + 4 * i, true);
        }
      } else if (meta.dtype === "F16" || meta.dtype === "BF16") {
        if (nbytes !== 2 * n) throw new Error("safetensors: size mismatch at " + name);
        data = new Float32Array(n);
        const bits = new Uint32Array(data.buffer);
        for (let i = 0; i < n; i++) {
          const h = dv.getUint16(start + 2 * i, true);
          if (meta.dtype === "BF16") bits[i] = h << 16;
          else data[i] = f16ToF32(h);
        }
      } else {
        throw new Error("safetensors: " + name + " is " + meta.dtype + "; this reader handles F32, F16 and BF16");
      }
      out[name] = { shape, data };
    }
    return out;
  }

  // ---- word tokenizer (word_tokenizer.hpp) --------------------------------
  const EOS_LITERAL = "<|endoftext|>";
  const EOS_BYTES = Array.from(EOS_LITERAL, c => c.charCodeAt(0));
  const isAlpha = c => (c >= 65 && c <= 90) || (c >= 97 && c <= 122);
  const isDigit = c => c >= 48 && c <= 57;
  const isSpace = c => c === 32 || (c >= 9 && c <= 13);

  class Vocab {
    constructor(tokens) {
      this.tokens = tokens.slice();
      this.map = new Map();
      // std::map::emplace keeps the FIRST id of a repeated string.
      tokens.forEach((t, i) => { if (!this.map.has(t)) this.map.set(t, i); });
      this.eos = this.map.has(EOS_LITERAL) ? this.map.get(EOS_LITERAL) : -1;
    }
    id(s) { const v = this.map.get(s); return v === undefined ? 0 : v; }

    // Byte-wise, as the C++ (C locale): ASCII letters, digits and apostrophes
    // form words (lower-cased); any other non-space byte is a token by itself.
    // A lone non-ASCII byte is never a vocabulary string here, so it is id 0.
    tokenize(text, maxTokens = Infinity) {
      const b = new TextEncoder().encode(text);
      const ids = [];
      let cur = "";
      const flush = () => { if (cur) { ids.push(this.id(cur)); cur = ""; } };
      for (let i = 0; i < b.length; i++) {
        if (ids.length >= maxTokens) break;
        const c = b[i];
        if (c === 60 && this.eos >= 0 && EOS_BYTES.every((x, k) => b[i + k] === x)) {
          flush();
          ids.push(this.eos);
          i += EOS_BYTES.length - 1;
        } else if (isAlpha(c) || c === 39 || isDigit(c)) {
          cur += String.fromCharCode(c >= 65 && c <= 90 ? c + 32 : c);
        } else {
          flush();
          if (!isSpace(c)) ids.push(c < 128 ? this.id(String.fromCharCode(c)) : 0);
        }
      }
      flush();
      return ids;
    }

    // Words space-separated; ".,!?;:)" attach to the word before; nothing
    // follows "(" with a space.
    detokenize(ids) {
      let out = "";
      for (const id of ids) {
        if (id < 0 || id >= this.tokens.length) continue;
        const t = this.tokens[id];
        const attach = t.length === 1 && ".,!?;:)".includes(t);
        if (out && !attach && !out.endsWith("(")) out += " ";
        out += t;
      }
      return out;
    }
  }

  function endsWith(ids, stop) {
    if (!stop.length || ids.length < stop.length) return false;
    for (let k = 0; k < stop.length; k++) if (ids[ids.length - stop.length + k] !== stop[k]) return false;
    return true;
  }

  // ---- kernels (one row at a time) -----------------------------------------
  // y[out] = x[in] @ W[in, out] (+ b)
  function linear(x, W, b, nin, nout, y) {
    const acc = new Float64Array(nout);
    for (let i = 0; i < nin; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * nout;
      for (let j = 0; j < nout; j++) acc[j] += xi * W[row + j];
    }
    for (let j = 0; j < nout; j++) y[j] = b ? f32(f32(acc[j]) + b[j]) : acc[j];
    return y;
  }

  function rmsnorm(x, w, d, y) {
    let ss = 0;
    for (let j = 0; j < d; j++) ss += x[j] * x[j];
    const inv = f32(1 / Math.sqrt(f32(f32(ss) / d) + f32(1e-5)));
    for (let j = 0; j < d; j++) y[j] = f32(x[j] * inv) * w[j];
    return y;
  }

  function layernorm(x, g, bta, d, y) {
    let mu = 0;
    for (let j = 0; j < d; j++) mu += x[j];
    mu = f32(f32(mu) / d);
    let v = 0;
    for (let j = 0; j < d; j++) { const t = f32(x[j] - mu); v += t * t; }
    const rs = f32(1 / Math.sqrt(f32(f32(v) / d) + f32(1e-5)));
    for (let j = 0; j < d; j++) y[j] = f32(g[j] * f32(f32(x[j] - mu) * rs)) + bta[j];
    return y;
  }

  const SQRT_2_OVER_PI = f32(0.7978845608028654);
  const gelu = v => f32(v * f32(0.5 * f32(1 + Math.tanh(f32(SQRT_2_OVER_PI * f32(v + f32(f32(f32(0.044715 * v) * v) * v)))))));
  const silu = v => f32(v / f32(1 + Math.exp(-v)));
  const sigmoid = v => f32(1 / f32(1 + Math.exp(-v)));

  // ---- the model -----------------------------------------------------------
  const SUPPORTED_ATTENTION = new Set(["exact"]);

  class Model {
    // manifest: model/manifest.json (tools/export_web_chat.py); tensors:
    // parseSafetensors() output.
    constructor(manifest, tensors) {
      const m = manifest;
      this.manifest = m;
      this.family = m.family;
      if (!["llama", "flex", "gpt2"].includes(this.family))
        throw new Error("family '" + this.family + "' is not supported in the browser (llama, flex, gpt2 are)");
      const attention = m.attention || "exact";
      if (!SUPPORTED_ATTENTION.has(attention))
        throw new Error("attention '" + attention + "' is not supported in the browser (exact is)");
      this.d = m.d; this.L = m.layers; this.H = m.heads; this.T = m.T;
      this.dk = this.d / this.H;
      this.ropeTheta = m.rope_theta || 10000;
      // Which heads RoPE rotates (spec arch.rope_heads). A manifest without
      // the field predates it and came from a "first" run.
      this.ropeHeads = m.rope_heads || "first";
      if (this.ropeHeads !== "all" && this.ropeHeads !== "first")
        throw new Error("rope_heads '" + this.ropeHeads + "' is not supported (all, first are)");
      const need = (name, n) => {
        const t = tensors[name];
        if (!t) throw new Error("weights are missing " + name);
        if (n !== undefined && t.data.length !== n) throw new Error(name + " has " + t.data.length + " values, expected " + n);
        return t.data;
      };
      const opt = name => (tensors[name] ? tensors[name].data : null);
      const d = this.d;
      // ParityLM names -> FlexLM names (same decoder, 2 blocks).
      const P = this.family === "gpt2"
        ? (b, s) => ({ ln_1: "ln1_" + b, ln_2: "ln2_" + b, attn: "attn_" + b, mlp: "mlp_" + b })[s]
        : (b, s) => "layers." + b + "." + s;

      if (this.family === "llama") {
        this.pos = "rope"; this.normKind = "rmsnorm"; this.act = "swiglu"; this.residual = "residual";
        this.wte = need("embed_tokens.weight");
        this.V = tensors["embed_tokens.weight"].shape[0];
        this.dff = tensors["layers.0.mlp.gate_proj.weight"].shape[1];
        this.blocks = [];
        for (let b = 0; b < this.L; b++) {
          const p = "layers." + b + ".";
          this.blocks.push({
            n1w: need(p + "input_layernorm.weight", d), n1b: null,
            n2w: need(p + "post_attention_layernorm.weight", d), n2b: null,
            q: need(p + "self_attn.q_proj.weight", d * d), k: need(p + "self_attn.k_proj.weight", d * d),
            v: need(p + "self_attn.v_proj.weight", d * d),
            o: need(p + "self_attn.o_proj.weight", d * d), ob: null,
            gate: need(p + "mlp.gate_proj.weight"), up: need(p + "mlp.up_proj.weight"),
            down: need(p + "mlp.down_proj.weight"),
          });
        }
        this.nfw = need("norm.weight", d); this.nfb = null;
        this.head = opt("lm_head.weight");  // [d, V] when untied
      } else {
        this.pos = this.family === "gpt2" ? "learned" : (m.position || "learned");
        this.normKind = this.family === "gpt2" ? "layernorm" : (m.norm || "layernorm");
        this.act = this.family === "gpt2" ? "gelu" : (m.activation || "gelu");
        this.residual = this.family === "gpt2" ? "residual" : (m.residual || "residual");
        if (this.pos === "rope") throw new Error("rope positions belong to the llama family");
        this.wte = need("wte.weight");
        this.V = tensors["wte.weight"].shape[0];
        if (this.pos === "learned") this.wpe = need("wpe.weight");
        else if (this.pos === "sinusoidal") {
          // FlexLM's fixed table: double angle, cast to float.
          this.wpe = new Float32Array(this.T * d);
          for (let p = 0; p < this.T; p++)
            for (let i = 0; i < d; i++) {
              const angle = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d);
              this.wpe[p * d + i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
            }
        } else throw new Error("unknown position encoding " + this.pos);
        this.blocks = [];
        for (let b = 0; b < this.L; b++) {
          const blk = {};
          if (this.normKind === "layernorm") {
            blk.n1w = need(P(b, "ln_1") + ".weight", d); blk.n1b = need(P(b, "ln_1") + ".bias", d);
            blk.n2w = need(P(b, "ln_2") + ".weight", d); blk.n2b = need(P(b, "ln_2") + ".bias", d);
          } else {
            blk.n1w = need(P(b, "norm1") + ".weight", d); blk.n1b = null;
            blk.n2w = need(P(b, "norm2") + ".weight", d); blk.n2b = null;
          }
          const a = P(b, "attn");
          blk.qkv = need(a + ".c_attn.weight", d * 3 * d); blk.qkvb = need(a + ".c_attn.bias", 3 * d);
          blk.o = need(a + ".c_proj.weight", d * d); blk.ob = need(a + ".c_proj.bias", d);
          const f = P(b, "mlp");
          if (this.act === "swiglu") {
            blk.gate = need(f + ".gate_proj.weight"); blk.up = need(f + ".up_proj.weight");
            blk.down = need(f + ".down_proj.weight");
          } else {
            blk.fc = need(f + ".c_fc.weight"); blk.fcb = need(f + ".c_fc.bias");
            blk.proj = need(f + ".c_proj.weight"); blk.projb = need(f + ".c_proj.bias");
          }
          if (this.residual === "highway") {
            blk.ga = need(P(b, "gate_attn") + ".weight"); blk.gab = need(P(b, "gate_attn") + ".bias");
            blk.gm = need(P(b, "gate_mlp") + ".weight"); blk.gmb = need(P(b, "gate_mlp") + ".bias");
          }
          this.blocks.push(blk);
        }
        this.dff = this.act === "swiglu"
          ? tensors[P(0, "mlp") + ".gate_proj.weight"].shape[1]
          : tensors[P(0, "mlp") + ".c_fc.weight"].shape[1];
        if (this.normKind === "layernorm") { this.nfw = need("ln_f.weight", d); this.nfb = need("ln_f.bias", d); }
        else { this.nfw = need("norm.weight", d); this.nfb = null; }
        this.head = need("head.weight");  // [d, V]
      }
      if (this.head && this.head.length !== d * this.V) throw new Error("head shape does not match the vocabulary");
      if (this.wpe && this.wpe.length < this.T * d) throw new Error("position table is shorter than T");
      this.params = Object.values(tensors).reduce((a, t) => a + t.data.length, 0);
      // RoPE frequencies, float as the engine computes them.
      this.invFreq = new Float32Array(this.dk / 2);
      for (let dim = 0; dim < this.dk; dim += 2)
        this.invFreq[dim / 2] = f32(1 / f32(Math.pow(this.ropeTheta, f32(dim / this.dk))));
      this.reset();
    }

    reset() {
      const n = this.T * this.d;
      this.K = this.blocks.map(() => new Float32Array(n));
      this.Vc = this.blocks.map(() => new Float32Array(n));
      this.cached = [];  // ids whose K/V rows are in the cache, positions 0..
    }

    norm(x, w, b, y) {
      return this.normKind === "layernorm" || b ? layernorm(x, w, b, this.d, y) : rmsnorm(x, w, this.d, y);
    }

    // ops::apply_rope: within each head, adjacent pairs (x[2j], x[2j+1])
    // rotated by pos * inv_freq; every head under rope_heads "all", head 0
    // only under "first" (the engine's legacy coverage).
    rope(vec, pos) {
      const nRot = this.ropeHeads === "all" ? this.H : 1;
      for (let dim = 0; dim < this.dk; dim += 2) {
        const th = f32(pos * this.invFreq[dim / 2]);
        const c = f32(Math.cos(th)), s = f32(Math.sin(th));
        for (let h = 0; h < nRot; h++) {
          const i = h * this.dk + dim;
          const x0 = vec[i], x1 = vec[i + 1];
          vec[i] = f32(x0 * c) - f32(x1 * s);
          vec[i + 1] = f32(x0 * s) + f32(x1 * c);
        }
      }
    }

    // One token at position `pos` (== this.cached.length); fills that row of
    // every layer's K/V cache. Returns logits when wantLogits.
    step(id, wantLogits) {
      const d = this.d, H = this.H, dk = this.dk, pos = this.cached.length;
      if (pos >= this.T) throw new Error("step past the context window");
      if (id < 0 || id >= this.V) throw new Error("token id out of range");
      const x = new Float32Array(d), h = new Float32Array(d);
      for (let j = 0; j < d; j++) x[j] = this.wte[id * d + j];
      if (this.wpe) for (let j = 0; j < d; j++) x[j] = x[j] + this.wpe[pos * d + j];
      const q = new Float32Array(d), k = new Float32Array(d), v = new Float32Array(d);
      const att = new Float32Array(d), ao = new Float32Array(d), x1 = new Float32Array(d);
      const sc = f32(1 / Math.sqrt(dk));
      const scores = new Float64Array(pos + 1);
      for (let b = 0; b < this.blocks.length; b++) {
        const B = this.blocks[b];
        this.norm(x, B.n1w, B.n1b, h);
        if (B.qkv) {
          const qkv = linear(h, B.qkv, B.qkvb, d, 3 * d, new Float32Array(3 * d));
          q.set(qkv.subarray(0, d)); k.set(qkv.subarray(d, 2 * d)); v.set(qkv.subarray(2 * d));
        } else {
          linear(h, B.q, null, d, d, q); linear(h, B.k, null, d, d, k); linear(h, B.v, null, d, d, v);
        }
        if (this.pos === "rope") { this.rope(q, pos); this.rope(k, pos); }
        const Kc = this.K[b], Vc = this.Vc[b];
        Kc.set(k, pos * d); Vc.set(v, pos * d);
        for (let hh = 0; hh < H; hh++) {
          const o = hh * dk;
          let mx = -1e30;
          for (let t = 0; t <= pos; t++) {
            let s = 0;
            for (let i = 0; i < dk; i++) s += q[o + i] * Kc[t * d + o + i];
            s = f32(f32(s) * sc);
            scores[t] = s;
            if (s > mx) mx = s;
          }
          let z = 0;
          for (let t = 0; t <= pos; t++) { scores[t] = f32(Math.exp(scores[t] - mx)); z += scores[t]; }
          z = f32(z);
          const acc = new Float64Array(dk);
          for (let t = 0; t <= pos; t++) {
            const w = f32(scores[t] / z);
            for (let i = 0; i < dk; i++) acc[i] += w * Vc[t * d + o + i];
          }
          for (let i = 0; i < dk; i++) att[o + i] = acc[i];
        }
        linear(att, B.o, B.ob, d, d, ao);
        this.combine(x, ao, B.ga, B.gab, x1);
        this.norm(x1, B.n2w, B.n2b, h);
        let f;
        if (this.act === "swiglu") {
          const g = linear(h, B.gate, null, d, this.dff, new Float32Array(this.dff));
          const u = linear(h, B.up, null, d, this.dff, new Float32Array(this.dff));
          for (let j = 0; j < this.dff; j++) g[j] = f32(silu(g[j]) * u[j]);
          f = linear(g, B.down, null, this.dff, d, new Float32Array(d));
        } else {
          const a = linear(h, B.fc, B.fcb, d, this.dff, new Float32Array(this.dff));
          for (let j = 0; j < this.dff; j++) a[j] = this.act === "gelu" ? gelu(a[j]) : Math.max(0, a[j]);
          f = linear(a, B.proj, B.projb, this.dff, d, new Float32Array(d));
        }
        this.combine(x1, f, B.gm, B.gmb, x);
      }
      this.cached.push(id);
      if (!wantLogits) return null;
      this.norm(x, this.nfw, this.nfb, h);
      const logits = new Float32Array(this.V);
      if (this.head) linear(h, this.head, null, d, this.V, logits);
      else {
        for (let t = 0; t < this.V; t++) {  // tied: h @ E^T
          let s = 0;
          const row = t * d;
          for (let j = 0; j < d; j++) s += h[j] * this.wte[row + j];
          logits[t] = s;
        }
      }
      return logits;
    }

    // Sublayer combine (FlexBlock::combine): residual, highway or plain.
    combine(x, f, g, gb, out) {
      const d = this.d;
      if (this.residual === "plain") { out.set(f); return out; }
      if (this.residual === "highway") {
        const gt = linear(x, g, gb, d, d, new Float32Array(d));
        for (let j = 0; j < d; j++) out[j] = x[j] + f32(sigmoid(gt[j]) * f32(f[j] - x[j]));
        return out;
      }
      for (let j = 0; j < d; j++) out[j] = x[j] + f[j];
      return out;
    }

    // Logits for the next token after `ids`, over the last T ids. Reuses the
    // cache when it holds a prefix of that window; otherwise recomputes it.
    logitsFor(ids) {
      const ctx = ids.length > this.T ? ids.slice(ids.length - this.T) : ids;
      if (!ctx.length) throw new Error("empty context");
      let keep = 0;
      while (keep < this.cached.length && keep < ctx.length && this.cached[keep] === ctx[keep]) keep++;
      if (keep === ctx.length) keep--;  // need the last token's logits
      this.cached.length = keep;        // K/V rows past `keep` are overwritten
      let logits = null;
      for (let i = keep; i < ctx.length; i++) logits = this.step(ctx[i], i === ctx.length - 1);
      return logits;
    }
  }

  // ---- sampler (LoadedLM::next) --------------------------------------------
  const BANNED = ["<unk>", "<s>", "</s>", "<pad>"];

  function pick(logits, vocab, { temperature = 0, topK = 40, topP = 1, random = Math.random } = {}) {
    const V = logits.length, inv = Math.max(temperature, 1e-4);
    const scored = new Float32Array(V);
    for (let j = 0; j < V; j++) scored[j] = logits[j] / f32(inv);
    for (const sp of BANNED) if (vocab.map.has(sp)) scored[vocab.map.get(sp)] = -1e30;
    const k0 = Math.min(Math.max(topK | 0, 1), V);
    if (temperature <= 0 || k0 === 1) {
      let best = 0;
      for (let j = 1; j < V; j++) if (scored[j] > scored[best]) best = j;
      return best;
    }
    const order = Array.from({ length: V }, (_, j) => j).sort((a, b) => scored[b] - scored[a] || a - b);
    let k = k0;
    const mx = scored[order[0]], p = new Float64Array(k);
    let z = 0;
    for (let j = 0; j < k; j++) z += (p[j] = Math.exp(scored[order[j]] - mx));
    if (topP > 0 && topP < 1) {
      let acc = 0;
      for (let j = 0; j < k; j++) if ((acc += p[j]) >= topP * z) { k = j + 1; break; }
      z = acc;
    }
    let r = random() * z;
    for (let j = 0; j < k; j++) if ((r -= p[j]) <= 0) return order[j];
    return order[k - 1];
  }

  // ---- chat (mtstudio chat) -------------------------------------------------
  // history: [{role: "user"|"assistant", content}] oldest first.
  function chatPrompt(history, userInput) {
    let p = "";
    for (const t of history || []) if (t.role === "user" || t.role === "assistant") p += t.role + ": " + t.content + " ";
    return p + "user: " + userInput + " assistant:";
  }

  function turnMarkers(vocab) {
    const out = [];
    for (const m of ["user:", "assistant:"]) {
      const seq = vocab.tokenize(m, 16);
      if (!seq.includes(0)) out.push(seq);
    }
    return out;
  }

  const nextTick = () => new Promise(r => setTimeout(r, 0));

  class Chat {
    constructor(model, vocab) {
      if (vocab.tokens.length !== model.V)
        throw new Error("vocabulary has " + vocab.tokens.length + " words but the weights expect " + model.V);
      this.model = model; this.vocab = vocab; this.markers = turnMarkers(vocab);
    }

    // Plain continuation, as `mtstudio sample`: stops only at end-of-text.
    continueIds(ids, nNew, opts = {}) {
      ids = ids.slice();
      const out = [];
      for (let t = 0; t < nNew; t++) {
        const id = pick(this.model.logitsFor(ids), this.vocab, opts);
        if (id === this.vocab.eos) break;
        ids.push(id); out.push(id);
      }
      return out;
    }

    // One chat reply, as POST /chat. onToken(partialReply) after each token.
    async reply(history, userInput, { maxNewTokens = 60, temperature = 0, topK = 40, topP = 0.9, random, onToken } = {}) {
      const prompt = chatPrompt(history, userInput);
      const ids = this.vocab.tokenize(prompt), out = [];
      let stop = "length";
      const opts = { temperature, topK, topP, random };
      for (let t = 0; t < maxNewTokens; t++) {
        const id = pick(this.model.logitsFor(ids), this.vocab, opts);
        if (id === this.vocab.eos) { stop = "eos"; break; }
        ids.push(id); out.push(id);
        const m = this.markers.find(mk => endsWith(out, mk));
        if (m) { out.length -= m.length; stop = "turn"; break; }
        if (onToken) onToken(this.vocab.detokenize(out).trim());
        await nextTick();
      }
      return { reply: this.vocab.detokenize(out).trim(), stop_reason: stop, tokens: out.length };
    }
  }

  // ---- loading ----------------------------------------------------------------
  // Browser: Paperkiln.load("model/") fetches manifest, weights and vocabulary
  // relative to the page. onProgress(bytesSoFar, bytesTotal) for the weights.
  async function load(base = "model/", onProgress) {
    const get = async (f, kind) => {
      const r = await fetch(base + f);
      if (!r.ok) throw new Error("cannot fetch " + base + f + " (" + r.status + ")");
      return kind === "json" ? r.json() : r;
    };
    const manifest = await get("manifest.json", "json");
    const vocabList = await get(manifest.vocab.file, "json");
    const r = await get(manifest.weights.file);
    const total = Number(r.headers.get("Content-Length")) || manifest.weights.bytes || 0;
    let buf;
    if (r.body && r.body.getReader && onProgress) {
      const reader = r.body.getReader(), parts = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value); got += value.length;
        onProgress(got, total);
      }
      const all = new Uint8Array(got);
      let o = 0;
      for (const p of parts) { all.set(p, o); o += p.length; }
      buf = all.buffer;
    } else buf = await r.arrayBuffer();
    if (manifest.weights.bytes && buf.byteLength !== manifest.weights.bytes)
      throw new Error("weights are " + buf.byteLength + " bytes, manifest says " + manifest.weights.bytes);
    return fromParts(manifest, buf, vocabList);
  }

  function fromParts(manifest, weightsBuffer, vocabList) {
    const vocab = new Vocab(vocabList);
    const model = new Model(manifest, parseSafetensors(weightsBuffer));
    return { manifest, vocab, model, chat: new Chat(model, vocab) };
  }

  return { parseSafetensors, Vocab, Model, Chat, pick, chatPrompt, endsWith, load, fromParts };
});
