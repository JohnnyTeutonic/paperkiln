"""Summaries of kvskip_measure.py checkpoints: certified-skip fractions, oracle fractions and the cost
model, by model, tiling, context bin, bound and certificate; per-head detail; JSON receipt.

    python kvskip_report.py OUT.json CKPT.npz [CKPT.npz ...]

Cost model (decode attention as reading K and V rows; one row = one key and its value, 2 * head_dim
floats): dense reads n rows. A certified kernel reads the own tile, the tiles it does not skip, and
each closed tile's metadata: norm 1 float, pairnorm head_dim/2, box 2 head_dim (Quest's min and max),
ball head_dim + 1, pairmin 2.5 head_dim; the "tile" certificate adds the tile's value min and max
(2 head_dim). The exact control reads every closed tile's keys (half a row per key) to get its bound,
so it saves value reads only. saving = 1 - rows / n, a share of keys read, not of wall time.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import kvskip_measure as M  # noqa: E402

HD = 64
META = {"norm": 1 / (2 * HD), "pairnorm": 0.25, "box": 1.0, "ball": (HD + 1) / (2 * HD), "pairmin": 1.25,
        "exact": 0.0}
VMETA = 1.0
BIN_LABEL = ["<=64", "65-128", "129-256", "257-384", "385-512"]


def load(path):
    z = np.load(path, allow_pickle=False)
    info = json.loads(str(z["done"]))
    return {k: z[k] for k in z.files if k != "done"}, info


def saving(st, sel, b, v):
    """Share of rows saved under bound b and certificate v over the cells selected (sum over them)."""
    keys = st["keys"][sel].sum()
    if keys == 0:
        return None
    sk = st["cert_keys"][..., b, v][sel]
    if (sk < 0).any():
        return None
    skipped = sk.sum()
    bname, vname = M.BOUNDS[b], M.VARIANTS[v]
    meta = META[bname] + (VMETA if vname.startswith("greedy_tile") else 0.0)
    tiles = st["tiles_elig"][sel].sum()
    extra = meta * tiles
    if bname == "exact":
        extra = 0.5 * st["keys_elig"][sel].sum() - 0.5 * skipped   # every closed key read (K only) to bound
    rows = keys - skipped + extra
    return {"skip_frac": skipped / keys, "saving": 1 - rows / keys}


def summarise(st):
    L, H, NT, NB = st["nq"].shape
    out = {}
    for ti, tl in enumerate(M.TILINGS):
        for bn in range(NB):
            sel = (slice(None), slice(None), ti, bn)
            keys = st["keys"][sel].sum()
            if keys == 0:
                continue
            tiles = st["tiles_elig"][sel].sum()
            cell = {"queries": float(st["nq"][sel].sum() / (L * H)), "keys": float(keys),
                    "closed_key_frac": float(st["keys_elig"][sel].sum() / keys),
                    "closed_tiles_per_query": float(tiles / st["nq"][sel].sum()),
                    "oracle_rounding_skip_frac": float(st["oracle_keys"][sel][..., 0].sum() / keys),
                    "oracle_halfulp_skip_frac": float(st["oracle_keys"][sel][..., 1].sum() / keys),
                    "tiles_safe_alone_frac": float(st["safe_single"][sel].sum() / max(tiles, 1)),
                    "tiles_mass_below": {f"{t:.3g}": float(st["mass_lt"][sel][..., i].sum() / max(tiles, 1))
                                         for i, t in enumerate(M.MASS_THR)},
                    "tiles_underflow_frac": {"denormal": float(st["underflow_tiles"][sel][..., 0].sum() / max(tiles, 1)),
                                             "ftz": float(st["underflow_tiles"][sel][..., 1].sum() / max(tiles, 1))},
                    "gap_hist": {f"{M.GAP_EDGES[i]}-{M.GAP_EDGES[i + 1]}": float(st["gap_hist"][sel][..., i].sum() / max(tiles, 1))
                                 for i in range(len(M.GAP_EDGES) - 1)},
                    "oracle_tailmass_skip_frac": {f"{t:g}": float(st["tail_keys"][sel][..., i].sum() / keys)
                                                  for i, t in enumerate(M.TAIL_MASS)},
                    "bound_minus_exact_nats": {b: float(st["loose"][sel][..., i].sum() / max(tiles, 1))
                                               for i, b in enumerate(M.BOUNDS)},
                    "tiles_bound_15_nats_under_max": {b: float(st["bound_gap15"][sel][..., i].sum() / max(tiles, 1))
                                                      for i, b in enumerate(M.BOUNDS)},
                    "mean_hull_range": float(st["hull_range"][sel].sum() / st["nq"][sel].sum()),
                    "max_abs_o": float(st["max_abs_o"][sel].max()),
                    "cert": {}}
            for b, bname in enumerate(M.BOUNDS):
                for v, vname in enumerate(M.VARIANTS):
                    r = saving(st, sel, b, v)
                    if r is not None:
                        cell["cert"][f"{bname}/{vname}"] = r
            out[f"{tl}/{BIN_LABEL[bn]}"] = cell
    return out


def per_head(st, ti, bn, b, v):
    L, H = st["nq"].shape[:2]
    rows = []
    for li in range(L):
        for h in range(H):
            sel = (li, h, ti, bn)
            keys = st["keys"][sel]
            if keys == 0:
                continue
            rows.append({"layer": li, "head": h,
                         "cert_skip_frac": float(st["cert_keys"][sel][b, v] / keys),
                         "oracle_skip_frac": float(st["oracle_keys"][sel][0] / keys),
                         "closed_key_frac": float(st["keys_elig"][sel] / keys)})
    return rows


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    rec = {}
    for p in paths:
        st, info = load(p)
        name = p.replace("\\", "/").rsplit("/", 1)[-1].replace(".npz", "")
        summ = summarise(st)
        best = {}
        for tl in M.TILINGS:
            bn = len(BIN_LABEL) - 1
            key = f"{tl}/{BIN_LABEL[bn]}"
            if key not in summ:
                continue
            c = summ[key]["cert"]
            k = max(c, key=lambda x: c[x]["saving"])
            best[tl] = {"best": k, **c[k]}
        rec[name] = {"contexts": info["extra"]["contexts"], "summary": summ, "best_at_385_512": best,
                     "per_head": {f"{tl}/{BIN_LABEL[bn]}/{bname}/{vname}": per_head(
                         st, M.TILINGS.index(tl), bn, M.BOUNDS.index(bname), M.VARIANTS.index(vname))
                         for tl in ("fixed32", "turn") for bn in (2, 4)
                         for bname, vname in (("pairmin", "greedy_tile_1e-5"), ("exact", "greedy_tile_1e-5"))}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    for name, r in rec.items():
        print("==", name, len(r["contexts"]), "contexts")
        for key, c in r["summary"].items():
            cb = c["cert"]
            pick = {k: round(cb[k]["skip_frac"], 3) for k in cb if k.split("/")[1] in ("greedy_tile_1e-5", "greedy_hull_1e-5")}
            print(f" {key:18s} closed {c['closed_key_frac']:.3f} oracle {c['oracle_rounding_skip_frac']:.3f} "
                  f"halfulp {c['oracle_halfulp_skip_frac']:.3f} uf {c['tiles_underflow_frac']['denormal']:.3f} "
                  f"m<1e-5 {c['tiles_mass_below']['1e-05']:.3f} |o|max {c['max_abs_o']:.2f}")
            print("    ", pick)


if __name__ == "__main__":
    main()
