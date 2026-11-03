"""Tests of compare_cuda.py on a synthetic tiny Llama (no GPU): python3 -m unittest test_compare_cuda -v"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_cuda  # noqa: E402  (imports the frozen C2 referee)


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class CompareCudaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        d, H, F, V = 8, 2, 16, 11
        self.tmp = Path(tempfile.mkdtemp())
        hf = self.tmp / "hf"
        hf.mkdir()
        w = {"model.embed_tokens.weight": rng.normal(size=(V, d)), "lm_head.weight": rng.normal(size=(V, d)),
             "model.norm.weight": 1 + 0.1 * rng.normal(size=d)}
        p = "model.layers.0."
        for k, shape in (("self_attn.q_proj.weight", (d, d)), ("self_attn.k_proj.weight", (d, d)),
                         ("self_attn.v_proj.weight", (d, d)), ("self_attn.o_proj.weight", (d, d)),
                         ("mlp.gate_proj.weight", (F, d)), ("mlp.up_proj.weight", (F, d)),
                         ("mlp.down_proj.weight", (d, F))):
            w[p + k] = 0.3 * rng.normal(size=shape)
        w[p + "input_layernorm.weight"] = 1 + 0.1 * rng.normal(size=d)
        w[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.normal(size=d)
        write_safetensors(hf / "model.safetensors", w)
        (hf / "config.json").write_text(json.dumps({
            "hidden_size": d, "num_attention_heads": H, "num_key_value_heads": H, "num_hidden_layers": 1,
            "intermediate_size": F, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "vocab_size": V,
            "tie_word_embeddings": False}))
        self.ids = [1, 4, 2, 9, 7, 3]
        (self.tmp / "ids.txt").write_text(" ".join(map(str, self.ids)))
        ref, _ = compare_cuda.c2.referee(hf, self.ids)
        self.ref = ref.astype("<f4")
        self.hf = hf

    def logits(self, name, a):
        p = self.tmp / name
        np.asarray(a, dtype="<f4").tofile(p)
        return str(p)

    def run_compare(self, cuda, repeat=None, cpu=None, pk=None):
        out = self.tmp / "r.json"
        r = subprocess.run([sys.executable, str(HERE / "compare_cuda.py"), str(self.hf), str(self.tmp / "ids.txt"),
                            cuda, repeat or cuda, cpu or self.logits("cpu.f32", self.ref),
                            pk or self.logits("pk.f32", self.ref), str(out)], capture_output=True, text=True)
        return r.returncode, json.loads(out.read_text())

    def test_float32_referee_logits_pass(self):
        rc, rep = self.run_compare(self.logits("cuda.f32", self.ref))
        self.assertEqual(rc, 0, rep)
        self.assertTrue(rep["pass"] and rep["repeat_bit_identical"])
        self.assertEqual(rep["gated"], ["coalfire_cuda", "coalfire_cpu", "paperkiln_cpu"])
        self.assertLess(rep["legs"]["coalfire_cuda"]["max_rel_vs_float64"], 1e-6)
        self.assertTrue(rep["referee"].endswith("c2_coalfire_parity_20261103/compare.py"))

    def test_error_over_the_bound_fails(self):
        bad = self.ref.copy()
        bad[2, 5] += 3e-4 * max(1.0, float(np.abs(self.ref[2]).max()))
        rc, rep = self.run_compare(self.logits("cuda.f32", bad))
        self.assertEqual(rc, 1)
        self.assertFalse(rep["legs"]["coalfire_cuda"]["pass"])
        self.assertGreater(rep["legs"]["coalfire_cuda"]["max_rel_vs_float64"], 1e-4)

    def test_wrong_size_or_non_finite_fails(self):
        rc, rep = self.run_compare(self.logits("cuda.f32", self.ref[:-1]))
        self.assertEqual(rc, 1)
        self.assertIn("expected", rep["legs"]["coalfire_cuda"]["reason"])
        nan = self.ref.copy()
        nan[0, 0] = np.nan
        rc, rep = self.run_compare(self.logits("cuda.f32", nan))
        self.assertEqual(rc, 1)
        self.assertFalse(rep["legs"]["coalfire_cuda"]["finite"])

    def test_a_failing_cpu_leg_fails_and_a_differing_repeat_is_reported_only(self):
        bad = self.ref.copy()
        bad[0] = -bad[0]
        rc, rep = self.run_compare(self.logits("cuda.f32", self.ref), cpu=self.logits("cpu_bad.f32", bad))
        self.assertEqual(rc, 1)
        self.assertFalse(rep["legs"]["coalfire_cpu"]["pass"])
        nudged = self.ref.copy()
        nudged[1, 1] = np.nextafter(nudged[1, 1], np.float32(np.inf), dtype=np.float32)
        rc, rep = self.run_compare(self.logits("cuda.f32", self.ref), repeat=self.logits("rep.f32", nudged))
        self.assertEqual(rc, 0, rep)
        self.assertFalse(rep["repeat_bit_identical"])
        signed = self.ref.copy()  # -0.0 against 0.0: equal as numbers, different bits
        signed[0, 0] = 0.0
        flipped = signed.copy()
        flipped[0, 0] = -0.0
        rc, rep = self.run_compare(self.logits("cuda.f32", signed), repeat=self.logits("rep.f32", flipped))
        self.assertFalse(rep["repeat_bit_identical"])


if __name__ == "__main__":
    unittest.main()
