"""The NOTE's tables from kvskip_measure.py checkpoints (markdown on stdout, compact JSON receipt).

    python kvskip_tables.py OUT.json CKPT.npz [CKPT.npz ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import kvskip_measure as M
import kvskip_report as R

KEYFREE = ["norm", "pairnorm", "box", "ball", "pairmin"]
BINSETS = {"<=128": [0, 1], "129-256": [2], "257-384": [3], "385-512": [4]}


def sel_of(ti, bins, layers=slice(None), heads=slice(None)):
    return layers, heads, ti, bins


def frac(st, field, sel, idx=None):
    a = st[field][sel]
    if idx is not None:
        a = a[..., idx]
    k = st["keys"][sel].sum()
    return float(a.sum() / k) if k else float("nan")


def best_keyfree(st, sel):
    best = None
    for b in KEYFREE:
        for v, vn in enumerate(M.VARIANTS):
            r = R.saving(st, sel, M.BOUNDS.index(b), v)
            if r is None:
                continue
            if best is None or r["skip_frac"] > best[2]["skip_frac"] or (
                    r["skip_frac"] == best[2]["skip_frac"] and r["saving"] > best[2]["saving"]):
                best = (b, vn, r)
    return best


def fmt_best(bk):
    if bk[2]["skip_frac"] == 0:
        return "0 (every bound)"
    return f"{bk[2]['skip_frac']:.4f} ({bk[0]}/{bk[1]})"


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rec = {}
    lines = []
    P = lambda s: lines.append(s)  # noqa: E731
    models = []
    for p in paths:
        st, info = R.load(p)
        name = p.replace("\\", "/").rsplit("/", 1)[-1].replace(".npz", "").removeprefix("ck_")
        models.append((name, st, info))
    # ---- table 1: headline
    P("| model | tiling | keys (n) | closed share | oracle ROUNDING | oracle half-ulp | underflow tiles | best key-free certified skip (bound/certificate) | its net saving | exact-max control skip |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for name, st, info in models:
        rec[name] = {"contexts": info["extra"]["contexts"], "cells": {}}
        for ti, tl in enumerate(M.TILINGS):
            for bl, bins in BINSETS.items():
                sel = sel_of(ti, bins)
                if st["keys"][sel].sum() == 0:
                    continue
                tiles = st["tiles_elig"][sel].sum()
                bk = best_keyfree(st, sel)
                ex = R.saving(st, sel, M.BOUNDS.index("exact"), M.VARIANTS.index("greedy_tile_1e-5")) or \
                    R.saving(st, sel, M.BOUNDS.index("exact"), M.VARIANTS.index("greedy_hull_1e-5"))
                cell = {"keys_mean": float(st["keys"][sel].sum() / st["nq"][sel].sum()),
                        "closed_share": frac(st, "keys_elig", sel),
                        "oracle_rounding": frac(st, "oracle_keys", sel, 0),
                        "oracle_halfulp": frac(st, "oracle_keys", sel, 1),
                        "underflow_tile_share": float(st["underflow_tiles"][sel][..., 0].sum() / max(tiles, 1)),
                        "underflow_ftz_tile_share": float(st["underflow_tiles"][sel][..., 1].sum() / max(tiles, 1)),
                        "tail_mass_skip": {f"{t:g}": frac(st, "tail_keys", sel, i) for i, t in enumerate(M.TAIL_MASS)},
                        "tiles_mass_below": {f"{t:.3g}": float(st["mass_lt"][sel][..., i].sum() / max(tiles, 1))
                                             for i, t in enumerate(M.MASS_THR)},
                        "gap_cdf": {f"<{M.GAP_EDGES[i + 1]:g}": float(st["gap_hist"][sel][..., :i + 1].sum() / max(tiles, 1))
                                    for i in range(len(M.GAP_EDGES) - 2)},
                        "bound_above_tile_max_nats": {b: float(st["loose"][sel][..., M.BOUNDS.index(b)].sum() / max(tiles, 1))
                                                      for b in KEYFREE},
                        "best_keyfree": {"bound": bk[0], "certificate": bk[1], **bk[2]} if bk else None,
                        "all_certs": {f"{b}/{vn}": R.saving(st, sel, M.BOUNDS.index(b), v)
                                      for b in M.BOUNDS for v, vn in enumerate(M.VARIANTS)},
                        "exact_control": ex}
                rec[name]["cells"][f"{tl}/{bl}"] = cell
                if bl in ("<=128", "129-256", "385-512"):
                    P(f"| {name} | {tl} | {bl} | {cell['closed_share']:.2f} | {cell['oracle_rounding']:.4f} | "
                      f"{cell['oracle_halfulp']:.4f} | {cell['underflow_tile_share']:.4f} | "
                      f"{fmt_best(bk)} | {bk[2]['saving']:+.3f} | {ex['skip_frac']:.4f} |")
    # ---- table 2: why: gaps, looseness, tail mass (fixed32 and turn, 385-512 or largest bin)
    P("")
    P("| model | tiling | bin | tiles within 5 / 10 / 15 nats of the row max | tiles with mass < 1e-5 | bound above the tile's true max, nats (norm / pairnorm / box / ball / pairmin) | keys skippable at tail mass 1e-4 / 1e-3 / 1e-2 / 5e-2 |")
    P("|---|---|---|---|---|---|---|")
    for name, st, info in models:
        for tl in ("fixed32", "turn"):
            for bl in ("129-256", "385-512"):
                c = rec[name]["cells"].get(f"{tl}/{bl}")
                if not c:
                    continue
                g = c["gap_cdf"]
                lo = c["bound_above_tile_max_nats"]
                tm = c["tail_mass_skip"]
                P(f"| {name} | {tl} | {bl} | {g['<5']:.2f} / {g['<10']:.2f} / {g['<15']:.3f} | {c['tiles_mass_below']['1e-05']:.4f} | "
                  f"{lo['norm']:.1f} / {lo['pairnorm']:.1f} / {lo['box']:.1f} / {lo['ball']:.1f} / {lo['pairmin']:.1f} | "
                  f"{tm['0.0001']:.3f} / {tm['0.001']:.3f} / {tm['0.01']:.3f} / {tm['0.05']:.3f} |")
    # ---- table 3: heads (fixed32, 385-512 where present else 257-384)
    P("")
    P("| model | bin | heads | oracle ROUNDING skip: median / 90th pct / max head | heads with oracle > 5% / > 20% | heads where any key-free certificate skips > 0.1% | oracle by depth (first / middle / last third of layers) |")
    P("|---|---|---|---|---|---|---|")
    for name, st, info in models:
        L, H = st["nq"].shape[:2]
        ti = M.TILINGS.index("fixed32")
        bl = "385-512" if st["keys"][:, :, ti, 4].sum() else "257-384"
        bins = BINSETS[bl]
        hv, cert_heads, ph = [], 0, []
        for li in range(L):
            for h in range(H):
                sel = (li, h, ti, bins)
                o = frac(st, "oracle_keys", sel, 0)
                hv.append(o)
                bk = best_keyfree(st, sel)
                c = bk[2]["skip_frac"] if bk else 0.0
                cert_heads += c > 1e-3
                tl_ = max(st["tiles_elig"][sel].sum(), 1)
                ph.append({"layer": li, "head": h, "oracle_rounding": o, "best_keyfree_skip": c,
                           "tail_1e-3": frac(st, "tail_keys", sel, 1),
                           "tiles_gap_ge_15": float(st["gap_hist"][sel][..., 4:].sum() / tl_),
                           "ball_above_max_nats": float(st["loose"][sel][..., M.BOUNDS.index("ball")].sum() / tl_),
                           "pairmin_above_max_nats": float(st["loose"][sel][..., M.BOUNDS.index("pairmin")].sum() / tl_)})
        hv = np.array(hv)
        thirds = [frac(st, "oracle_keys", (slice(a, b), slice(None), ti, bins), 0)
                  for a, b in ((0, L // 3), (L // 3, 2 * L // 3), (2 * L // 3, L))]
        rec[name]["per_head_fixed32"] = {"bin": bl, "heads": ph}
        P(f"| {name} | {bl} | {L}x{H} | {np.median(hv):.4f} / {np.quantile(hv, 0.9):.4f} / {hv.max():.4f} | "
          f"{(hv > 0.05).sum()} / {(hv > 0.2).sum()} | {cert_heads} | {thirds[0]:.4f} / {thirds[1]:.4f} / {thirds[2]:.4f} |")
    P("")
    P("Most local heads (fixed32): layer.head, oracle skip, tiles >= 15 nats under the row max, ball / pairmin above the tile max (nats), best key-free skip")
    for name, st, info in models:
        hs = sorted(rec[name]["per_head_fixed32"]["heads"], key=lambda r: -r["oracle_rounding"])[:5]
        P(f"- {name}: " + "; ".join(f"{r['layer']}.{r['head']} {r['oracle_rounding']:.3f}, {r['tiles_gap_ge_15']:.3f}, "
                                   f"{r['ball_above_max_nats']:.1f} / {r['pairmin_above_max_nats']:.1f}, {r['best_keyfree_skip']:.4f}" for r in hs))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
