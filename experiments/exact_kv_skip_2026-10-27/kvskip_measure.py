"""Exactness-certified KV tile skipping: the measurement (microtorch/experiments/exact_kv_skip_2026-10-27).

Runs real chat contexts through a pick with the keramos numpy IR reference (spec/tools/ir_numpy.py),
captures every layer's post-RoPE q, k and v, and at every query position (= a decode step: causal
attention at p sees keys 0..p, so one prefill gives every decode step of the context), layer, head and
tiling records:
  (a) each closed tile's softmax mass;
  (b) oracle safety: tiles whose mass is below fixed thresholds, tiles whose omission alone keeps the
      output within ROUNDING, the largest ascending-mass skip set within ROUNDING (exact error vector),
      the same within half an f32 ulp of every output coordinate, and tiles whose every weight
      underflows to exactly 0 in f32 (the order-free bit-exact regime);
  (c) certified skips from key-free upper bounds on q.k (norm, pair-norm, box = Quest, ball, pairmin,
      plus the key-reading exact maximum as a control), under three certificates (below);
  (d) keys skipped and tiles bounded, for the cost model applied in kvskip_report.py.

Certificate (o = dense output, S = tiles read, T = tiles skipped, delta = T's dense mass):
  o - o_S = delta (mu_T - o_S),  delta = Z_T / (Z_S + Z_T) <= Z_T / Z_S <= sum_t n_t exp(U_t - m) / Z_S
  so |o_i - o_S,i| <= sum_t c_t R_t,i  with c_t = n_t exp(U_t - m) / Z_S and R_t,i any bound on
  |mu_t,i - o_S,i|:  "hull": the per-dim range of all values up to p (one running min/max per head);
  "tile": max(vmax_t,i - o_S,i, o_S,i - vmin_t,i) (per-tile value min/max).
  Skip when max_i bound <= tol * max(1, max_i(|o_S,i| - bound_i)) (a lower bound on the reference's
  max(1, max|ref|), IR_SPEC 4.3).
  greedy: read the own tile, then tiles by descending U until the rest certify.
  twophase: read the own tile and the top-1 tile, then skip the lowest-U tiles whose summed bound fits.

    python kvskip_measure.py MODEL --ckpt CKPT.npz [--contexts N] [--layers 0,1,..]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = "C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch"
sys.path.insert(0, ROOT + "/spec/tools")
sys.path.insert(0, ROOT + "/tools")
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402
import bpe  # noqa: E402
import chat_template  # noqa: E402

MODELS = {
    "pick_e1_L16": "C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf",
    "pick_e1_60m_L32": "C:/ml_artifacts/tiny_chat/e1_60m/pick_e1_60m_L32/model.gguf",
    "pick_e5_s0": "C:/ml_artifacts/tiny_chat/e5/pick_e5_s0/model.gguf",
    "pk_sft_v1": "C:/ml_artifacts/general_chat/sft/pk_sft_v1.gguf",
}
CARRY12 = ROOT + "/corpus/eval/carry12_v1.jsonl"
DIALOGUE = {"dailydialog": ROOT + "/corpus/data/human_dailydialog_v1/dialogues.jsonl",
            "bst": ROOT + "/corpus/data/human_bst_v1/dialogues.jsonl"}
LEVER1 = "C:/ml_artifacts/next_base/lever1_conv/lever1_conv_v1.jsonl"

TILINGS = ["fixed32", "fixed64", "fixed128", "turn", "message"]
BINS = [(0, 64), (64, 128), (128, 256), (256, 384), (384, 512)]   # number of keys n = p + 1, (lo, hi]
BOUNDS = ["norm", "pairnorm", "box", "ball", "pairmin", "exact"]
VARIANTS = ["greedy_hull_1e-5", "greedy_hull_1e-6", "greedy_tile_1e-5", "twophase_hull_1e-5"]
TILE_VARIANT_BOUNDS = {"box", "pairmin", "exact"}
TILE_VARIANT_TILINGS = {"fixed32", "fixed64", "fixed128", "turn"}
MASS_THR = [2.0 ** -24, 1e-7, 1e-6, 1e-5, 1e-4]
GAP_EDGES = [0, 5, 10, 15, 20, 30, 50, 87.34, 103.97, np.inf]   # nats below the row max
TOL = 1e-5
TAIL_MASS = [1e-4, 1e-3, 1e-2, 5e-2]   # oracle tail-mass budgets (approximate regime, for contrast)
UF_DENORM, UF_FTZ = 103.97, 87.34   # exp(-x) rounds to +0 in f32 (with denormals; flush-to-zero)
EOT, USER, ASSISTANT = 1, 2, 3


# ---------------------------------------------------------------- contexts
def render(messages):
    msgs = [m for m in messages if m.get("content")]
    while msgs and msgs[-1]["role"] != "assistant":
        msgs = msgs[:-1]
    if len(msgs) < 2:
        return None
    try:
        return chat_template.render(msgs)
    except Exception:  # reserved text inside content, a structure the template refuses
        return None


def carry12_text(row):
    return "".join("<|user|>" + t["user"] + "<|assistant|>" + t["gold"] + "<|endoftext|>" for t in row["turns"])


def tail_rows(path, nbytes):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        buf = f.read()
    lines = buf.split(b"\n")[1:]
    return [json.loads(x) for x in lines if x.strip()]


def build_contexts(tok, n_each=3, T=512):
    """(cid, source, ids) with len(ids) <= T. Carry-12: four single conversations (all shorter than
    T) and two chained pairs (24 exchanges, cut at T). Human dialogue: the last dialogues of
    DailyDialog, BST and EmpatheticDialogues (corpus tails) and the tail of lever1_conv_v1 (pk_sft_v1's
    SFT file; mtstudio holds out a file's tail), the longest first."""
    out = []
    rows = [json.loads(x) for x in open(CARRY12, encoding="utf-8")]
    for i in (0, 400, 800, 1200):
        out.append((rows[i]["id"], "carry12", tok.encode(carry12_text(rows[i]))[:T]))
    for i in (100, 900):
        out.append((rows[i]["id"] + "+next", "carry12x2",
                    tok.encode(carry12_text(rows[i]) + carry12_text(rows[i + 1]))[:T]))
    for name, path in DIALOGUE.items():
        rs = [json.loads(x) for x in open(path, encoding="utf-8")][-400:]
        cands = []
        for j, r in enumerate(rs):
            t = render(r["messages"])
            if t:
                cands.append((len(t), j, t))
        cands.sort(reverse=True)
        for _, j, t in cands[:n_each]:
            out.append((f"{name}:tail{j}", "human", tok.encode(t)[:T]))
    rs = tail_rows(LEVER1, 3_000_000)
    cands = []
    for j, r in enumerate(rs):
        t = render(r["messages"])
        if t:
            cands.append((len(t), j, t))
    cands.sort(reverse=True)
    for _, j, t in cands[:n_each]:
        out.append((f"lever1:tail{j}", "lever1_tail", tok.encode(t)[:T]))
    return out


# ---------------------------------------------------------------- model
def load(path):
    meta, tensors, ds, raw, _ = ir_check.read_gguf(path)
    doc = json.loads(meta["paperkiln.ir"][1])
    W = {}
    for name, dims, typ, off in tensors:
        assert typ == 0, name
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=ds + off).reshape(list(reversed(dims)))
    return doc, W


def forward_capture(doc, W, ids, attn_override=None):
    """ir_numpy.forward with the block split at its attention node: returns (logits, [(q, k, v, a)] per
    layer). attn_override(i, q, k, v, node) -> a replaces the reference attention (sparse runs)."""
    env = {"tokens": np.array(ids), "pos": np.arange(len(ids))}
    ir_numpy.run_nodes(doc["embed"]["nodes"], env, W)
    nodes = doc["block"]["nodes"]
    ia = [n["op"] for n in nodes].index("attention")
    pre, att, post = nodes[:ia], nodes[ia], nodes[ia + 1:]
    caps = []
    for i in range(doc["block"]["repeat"]):
        wi = ir_numpy.weight_index(doc, i)
        ir_numpy.run_nodes(pre, env, W, i, wi=wi)
        q, k, v = (env[n] for n in att["in"])
        if attn_override is None:
            ir_numpy.run_nodes([att], env, W, i, wi=wi)
        else:
            env[att["out"]] = attn_override(i, q, k, v, att).astype(np.float32)
        caps.append((q.copy(), k.copy(), v.copy(), env[att["out"]].copy()))
        ir_numpy.run_nodes(post, env, W, i, wi=wi)
    ir_numpy.run_nodes(doc["head"]["nodes"], env, W)
    return env["logits"], caps


# ---------------------------------------------------------------- tilings
def tile_index(ids, tiling):
    ids = np.asarray(ids)
    T = len(ids)
    j = np.arange(T)
    if tiling.startswith("fixed"):
        return j // int(tiling[5:])
    if tiling == "turn":   # an exchange ends with its <|endoftext|>, which belongs to it
        eot = (ids == EOT).astype(np.int64)
        return np.concatenate([[0], np.cumsum(eot)[:-1]])
    if tiling == "message":   # each <|user|> / <|assistant|> opens a tile
        op = np.isin(ids, [USER, ASSISTANT]).astype(np.int64)
        op[0] = 1
        return np.cumsum(op) - 1
    raise ValueError(tiling)


# ---------------------------------------------------------------- one head
def bounds_for(qh, kh, tid, nt, S):
    """U[bound] [P, nt]: upper bounds on the scaled score of every key of tile t, for every query.
    Tiles are contiguous runs of positions (tid non-decreasing)."""
    P, hd = qh.shape
    sc = 0.125
    q = qh.astype(np.float64)
    k = kh.astype(np.float64)
    st = np.searchsorted(tid, np.arange(nt))
    kmin = np.minimum.reduceat(k, st, axis=0)
    kmax = np.maximum.reduceat(k, st, axis=0)
    kmaxn = np.maximum.reduceat(np.linalg.norm(k, axis=1), st)
    kp = np.sqrt(k[:, 0::2] ** 2 + k[:, 1::2] ** 2)   # interleaved RoPE pairs (x[2j], x[2j+1])
    kpm = np.maximum.reduceat(kp, st, axis=0)
    cnt = np.diff(np.append(st, len(tid))).astype(np.float64)
    cen = np.add.reduceat(k, st, axis=0) / cnt[:, None]
    rad = np.maximum.reduceat(np.linalg.norm(k - cen[tid], axis=1), st)
    qn = np.linalg.norm(q, axis=1)
    qp = np.sqrt(q[:, 0::2] ** 2 + q[:, 1::2] ** 2)
    qpos, qneg = np.maximum(q, 0), np.minimum(q, 0)
    U = {}
    U["norm"] = sc * qn[:, None] * kmaxn[None, :]
    U["pairnorm"] = sc * qp @ kpm.T
    U["box"] = sc * (qpos @ kmax.T + qneg @ kmin.T)
    U["ball"] = sc * (q @ cen.T + qn[:, None] * rad[None, :])
    boxpair = (np.maximum(q[:, None, :] * kmin[None], q[:, None, :] * kmax[None]))   # [P, nt, hd]
    boxpair = boxpair[:, :, 0::2] + boxpair[:, :, 1::2]
    U["pairmin"] = sc * np.minimum(boxpair, qp[:, None, :] * kpm[None]).sum(-1)
    U["exact"] = np.maximum.reduceat(S, st, axis=1)
    return U, cnt, st


def analyse_head(qh, kh, vh, tid, bins_of_p, stats, li, hi, ti, tile_variant=True):
    P, hd = qh.shape
    nt = int(tid.max()) + 1
    S32 = (qh @ kh.T) * np.float32(0.125)
    S = S32.astype(np.float64)
    causal = np.arange(P)[None, :] > np.arange(P)[:, None]
    S[causal] = -np.inf
    m = S.max(axis=1, keepdims=True)
    E = np.exp(S - m)                       # [P, P], 0 for masked
    Z = E.sum(axis=1)
    v = vh.astype(np.float64)
    o = (E @ v) / Z[:, None]                # dense output, f64
    # per (query, tile): mass and partial outputs
    onehot = np.zeros((P, nt))
    onehot[np.arange(P), tid] = 1.0
    Zt = E @ onehot                         # [P, nt]
    stt = np.searchsorted(tid, np.arange(nt))
    ent = np.append(stt[1:], P)
    Yt = np.stack([E[:, a:b] @ v[a:b] for a, b in zip(stt, ent)], 1)   # [P, nt, hd] sum of e*v per tile
    own = tid                               # the query's own tile index (tid[p])
    tt = np.arange(nt)[None, :]
    elig = tt < own[:, None]                # closed tiles before the own one
    U, cnt, _ = bounds_for(qh, kh, tid, nt, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        ex_max = U["exact"]
    gap = m - ex_max                        # >= 0
    # value metadata
    vmin_t = np.minimum.reduceat(v, stt, axis=0)
    vmax_t = np.maximum.reduceat(v, stt, axis=0)
    vrun_max = np.maximum.accumulate(v, axis=0)
    vrun_min = np.minimum.accumulate(v, axis=0)
    Rg = (vrun_max - vrun_min).max(axis=1)  # [P] hull range (max over dims)
    ref_scale = np.maximum(1.0, np.abs(o).max(axis=1))
    delta = Zt / Z[:, None]                 # tile masses
    n_keys = np.arange(P) + 1
    keys_elig = (elig * cnt[None, :]).sum(1)
    # ---- (b) oracle
    mu = Yt / np.where(Zt > 0, Zt, 1)[:, :, None]
    err_single = (delta / np.maximum(1 - delta, 1e-300))[:, :, None] * np.abs(mu - o[:, None, :])
    safe_single = (err_single.max(-1) <= TOL * ref_scale[:, None]) & elig
    dsort = np.where(elig, delta, np.inf)
    order = np.argsort(dsort, axis=1, kind="stable")
    d_s = np.take_along_axis(np.where(elig, delta, 0.0), order, 1)
    Y_s = np.take_along_axis(np.where(elig[:, :, None], Yt, 0.0), order[:, :, None], 1) / Z[:, None, None]
    n_s = np.take_along_axis(np.where(elig, cnt[None, :], 0.0), order, 1)
    cd = np.cumsum(d_s, 1)
    cY = np.cumsum(Y_s, 1)
    e = (cY - cd[:, :, None] * o[:, None, :]) / np.maximum(1 - cd, 1e-300)[:, :, None]
    nel = elig.sum(1)
    kk = np.arange(nt)[None, :]
    valid_k = kk < nel[:, None]
    ok_tol = (np.abs(e).max(-1) <= TOL * ref_scale[:, None]) & valid_k
    ulp = np.spacing(np.abs(o).astype(np.float32)).astype(np.float64) / 2
    ok_ulp = (np.abs(e) <= ulp[:, None, :]).all(-1) & valid_k

    def prefix_keys(ok):
        # largest k with every prefix 1..k ok (first failure stops)
        run = np.cumprod(ok, axis=1)
        nk = run.sum(1)
        cn = np.concatenate([np.zeros((P, 1)), np.cumsum(n_s, 1)], 1)
        return cn[np.arange(P), nk]
    or_tol = prefix_keys(ok_tol)
    # approximate regime for contrast: the ascending-mass prefix whose total mass stays under a budget
    tail_keys = np.stack([prefix_keys((cd <= tm) & valid_k) for tm in TAIL_MASS], 1)
    or_ulp = prefix_keys(ok_ulp)
    uf_d = ((gap > UF_DENORM) & elig).sum(1)
    uf_f = ((gap > UF_FTZ) & elig).sum(1)
    gh = np.zeros((P, len(GAP_EDGES) - 1))
    for b in range(len(GAP_EDGES) - 1):
        gh[:, b] = ((gap >= GAP_EDGES[b]) & (gap < GAP_EDGES[b + 1]) & elig).sum(1)
    mass_lt = np.stack([((delta < thr) & elig).sum(1) for thr in MASS_THR], 1)
    # ---- (c) certificates
    own_Z = Zt[np.arange(P), own]
    own_Y = Yt[np.arange(P), own]
    cert = np.zeros((P, len(BOUNDS), len(VARIANTS)))
    fin = np.isfinite(U["exact"]) & elig
    loose = np.stack([np.where(fin, U[b] - U["exact"], 0.0).sum(1) for b in BOUNDS], 1)          # nats
    bgap = np.stack([(np.where(fin, m - U[b], -np.inf) > 15.0).sum(1) for b in BOUNDS], 1)   # bound 15 nats under max
    for bi, bname in enumerate(BOUNDS):
        Ub = U[bname]
        ub = np.where(elig, cnt[None, :] * np.exp(np.minimum(Ub - m, 700.0)), 0.0)   # n_t e^{U_t - m}
        ordd = np.argsort(np.where(elig, -Ub, np.inf), axis=1, kind="stable")       # read order
        ub_s = np.take_along_axis(ub, ordd, 1)
        Z_s = np.take_along_axis(np.where(elig, Zt, 0.0), ordd, 1)
        Y_s2 = np.take_along_axis(np.where(elig[:, :, None], Yt, 0.0), ordd[:, :, None], 1)
        n_s2 = np.take_along_axis(np.where(elig, cnt[None, :], 0.0), ordd, 1)
        ZS = own_Z[:, None] + np.concatenate([np.zeros((P, 1)), np.cumsum(Z_s, 1)], 1)       # [P, nt+1]
        YS = own_Y[:, None, :] + np.concatenate([np.zeros((P, 1, hd)), np.cumsum(Y_s2, 1)], 1)
        oS = YS / ZS[:, :, None]
        rest = ub_s.sum(1, keepdims=True) - np.concatenate([np.zeros((P, 1)), np.cumsum(ub_s, 1)], 1)
        rest = np.maximum(rest, 0.0)
        dbar = rest / ZS                                                              # [P, nt+1]
        nread_el = np.concatenate([np.zeros((P, 1)), np.cumsum(n_s2, 1)], 1)
        kvalid = np.arange(nt + 1)[None, :] <= nel[:, None]

        for vi, vname in enumerate(VARIANTS):
            tol = 1e-6 if "1e-6" in vname else TOL
            if vname.startswith("greedy_hull"):
                B = dbar * Rg[:, None]
                scale = np.maximum(1.0, (np.abs(oS) - B[:, :, None]).max(-1))
                ok = (B <= tol * scale) & kvalid
                ok[np.arange(P), nel] = True        # everything read: exact
                kst = ok.argmax(1)
                cert[:, bi, vi] = keys_elig - nread_el[np.arange(P), kst]
            elif vname.startswith("greedy_tile"):
                if bname not in TILE_VARIANT_BOUNDS or not tile_variant:
                    cert[:, bi, vi] = np.nan
                    continue
                c_s = ub_s / 1.0                    # [P, nt] in read order; c_t = ub_t / ZS(k)
                vmax_s = vmax_t[ordd]               # [P, nt, hd]
                vmin_s = vmin_t[ordd]
                kst = np.full(P, -1)
                for k in range(nt + 1):
                    unread = (np.arange(nt)[None, :] >= k) & (np.arange(nt)[None, :] < nel[:, None])
                    o_k = oS[:, k, :]
                    R = np.maximum(vmax_s - o_k[:, None, :], o_k[:, None, :] - vmin_s)    # [P, nt, hd]
                    Bi = (np.where(unread, c_s, 0.0)[:, :, None] * np.where(unread[:, :, None], R, 0.0)).sum(1)
                    Bi = Bi / ZS[:, k][:, None]
                    scale = np.maximum(1.0, (np.abs(o_k) - Bi).max(-1))
                    okk = ((Bi.max(-1) <= tol * scale) | (k >= nel)) & (kst < 0) & (k <= nel)
                    kst[okk] = k
                cert[:, bi, vi] = keys_elig - nread_el[np.arange(P), kst]
            else:   # twophase: read own + top-1, then skip the lowest-U tiles that fit
                k0 = np.minimum(1, nel)
                Z0 = ZS[np.arange(P), k0]
                o0 = oS[np.arange(P), k0]
                # remaining tiles (read-order positions k0..nel-1), ascending U = reverse read order
                skipped = np.zeros(P)
                budget_c = np.zeros(P)
                for k in range(nt - 1, -1, -1):     # from the lowest U upwards
                    live = (k >= k0) & (k < nel)
                    add = ub_s[:, k] / Z0 * Rg
                    fits = live & (budget_c + add <= tol * np.maximum(1.0, np.abs(o0).max(-1) - (budget_c + add)))
                    stop = live & ~fits
                    budget_c = np.where(fits, budget_c + add, budget_c)
                    skipped = np.where(fits, skipped + n_s2[:, k], skipped)
                    k0 = np.where(stop, nt + 5, k0)  # once one fails, stop skipping (a prefix only)
                cert[:, bi, vi] = skipped
    # ---- accumulate by bin
    for bn, (lo, hi_) in enumerate(BINS):
        sel = (n_keys > lo) & (n_keys <= hi_)
        if not sel.any():
            continue
        key = (li, hi, ti, bn)
        stats["nq"][key] += sel.sum()
        stats["keys"][key] += n_keys[sel].sum()
        stats["keys_elig"][key] += keys_elig[sel].sum()
        stats["tiles_elig"][key] += nel[sel].sum()
        stats["mass_lt"][key] += mass_lt[sel].sum(0)
        stats["safe_single"][key] += safe_single[sel].sum()
        stats["oracle_keys"][key] += [or_tol[sel].sum(), or_ulp[sel].sum()]
        stats["tail_keys"][key] += tail_keys[sel].sum(0)
        stats["underflow_tiles"][key] += [uf_d[sel].sum(), uf_f[sel].sum()]
        stats["gap_hist"][key] += gh[sel].sum(0)
        stats["cert_keys"][key] += np.nan_to_num(cert[sel].sum(0), nan=-1.0)
        stats["max_abs_o"][key] = max(stats["max_abs_o"][key], float(np.abs(o[sel]).max()))
        stats["hull_range"][key] += Rg[sel].sum()
        stats["loose"][key] += loose[sel].sum(0)
        stats["bound_gap15"][key] += bgap[sel].sum(0)


def new_stats(L, H):
    sh = (L, H, len(TILINGS), len(BINS))
    return {"nq": np.zeros(sh), "keys": np.zeros(sh), "keys_elig": np.zeros(sh), "tiles_elig": np.zeros(sh),
            "mass_lt": np.zeros(sh + (len(MASS_THR),)), "safe_single": np.zeros(sh),
            "oracle_keys": np.zeros(sh + (2,)), "tail_keys": np.zeros(sh + (len(TAIL_MASS),)), "underflow_tiles": np.zeros(sh + (2,)),
            "gap_hist": np.zeros(sh + (len(GAP_EDGES) - 1,)),
            "cert_keys": np.zeros(sh + (len(BOUNDS), len(VARIANTS))), "max_abs_o": np.zeros(sh),
            "hull_range": np.zeros(sh), "loose": np.zeros(sh + (len(BOUNDS),)),
            "bound_gap15": np.zeros(sh + (len(BOUNDS),))}


def save_atomic(path, stats, done, extra):
    tmp = path + ".tmp.npz"
    np.savez(tmp, done=np.array(json.dumps({"done": done, "extra": extra})), **stats)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--contexts", type=int, default=0, help="first N contexts only (0 = all)")
    ap.add_argument("--layers", type=str, default="", help="comma list of layers (default all)")
    a = ap.parse_args()
    path = MODELS[a.model]
    doc, W = load(path)
    tok = bpe.load_vocab(path)
    ctxs = build_contexts(tok)
    if a.contexts:
        ctxs = ctxs[:a.contexts]
    att = [n for n in doc["block"]["nodes"] if n["op"] == "attention"][0]
    L, H, hd = doc["block"]["repeat"], att["heads"], att["head_dim"]
    assert att["kv_heads"] == H and att["scale"] == 0.125 and not att["window"]
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(L))
    if os.path.exists(a.ckpt):
        z = np.load(a.ckpt, allow_pickle=False)
        info = json.loads(str(z["done"]))
        done, extra = info["done"], info["extra"]
        stats = {k: z[k] for k in z.files if k != "done"}
    else:
        done, extra = [], {"contexts": []}
        stats = new_stats(L, H)
    last = time.time()
    for cid, src, ids in ctxs:
        if cid in done:
            continue
        t0 = time.time()
        logits, caps = forward_capture(doc, W, ids)
        tids = [tile_index(ids, t) for t in TILINGS]
        for li in layers:
            if f"{cid}|{li}" in done:   # a layer finished before a restart
                continue
            q, k, v, _ = caps[li]
            for h in range(H):
                sl = slice(h * hd, (h + 1) * hd)
                for ti, tid in enumerate(tids):
                    analyse_head(q[:, sl], k[:, sl], v[:, sl], tid, None, stats, li, h, ti,
                                 tile_variant=TILINGS[ti] in TILE_VARIANT_TILINGS)
            done.append(f"{cid}|{li}")
            if time.time() - last > 150:   # checkpoint about every 2.5 minutes
                save_atomic(a.ckpt, stats, done, extra)
                last = time.time()
        done.append(cid)
        extra["contexts"].append({"id": cid, "source": src, "tokens": len(ids),
                                  "turns": int(np.sum(np.asarray(ids) == EOT)), "secs": round(time.time() - t0, 1)})
        print(json.dumps(extra["contexts"][-1]), flush=True)
        save_atomic(a.ckpt, stats, done, extra)
        last = time.time()
    print("done", a.model, len(done))


if __name__ == "__main__":
    main()
