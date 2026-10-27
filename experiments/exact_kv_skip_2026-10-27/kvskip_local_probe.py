"""Follow-up probe: can key-free bounds certify skips on the most local heads if tiles shrink?

For each model, the five heads with the largest oracle ROUNDING skip share (fixed32, from the main
receipt), on the 512-token contexts, tilings fixed4 / fixed8 / fixed16 / fixed32: oracle skip share,
certified skip share and net saving per bound (hull certificate; tile certificate from 16 up).

    python kvskip_local_probe.py OUT.json TABLES.json MODEL [MODEL ...]
"""
from __future__ import annotations

import json
import sys

import kvskip_measure as M

PROBE_TILINGS = ["fixed4", "fixed8", "fixed16", "fixed32"]
HD = 64


def main():
    out, tables, models = sys.argv[1], sys.argv[2], sys.argv[3:]
    tab = json.load(open(tables))
    M.TILINGS = PROBE_TILINGS          # new_stats and the tiling index use this list
    res = {}
    for name in models:
        heads = sorted(tab[name]["per_head_fixed32"]["heads"], key=lambda r: -r["oracle_rounding"])[:5]
        doc, W = M.load(M.MODELS[name])
        tok = M.bpe.load_vocab(M.MODELS[name])
        ctxs = [c for c in M.build_contexts(tok) if len(c[2]) == 512]
        att = [n for n in doc["block"]["nodes"] if n["op"] == "attention"][0]
        L, H = doc["block"]["repeat"], att["heads"]
        st = M.new_stats(L, H)
        for cid, src, ids in ctxs:
            _, caps = M.forward_capture(doc, W, ids)
            for r in heads:
                li, h = r["layer"], r["head"]
                q, k, v, _ = caps[li]
                sl = slice(h * HD, (h + 1) * HD)
                for ti, tl in enumerate(PROBE_TILINGS):
                    M.analyse_head(q[:, sl], k[:, sl], v[:, sl], M.tile_index(ids, tl), None, st, li, h, ti,
                                   tile_variant=tl in ("fixed16", "fixed32"))
            print(name, cid, flush=True)
        meta = {"norm": 1 / (2 * HD), "pairnorm": 0.25, "box": 1.0, "ball": (HD + 1) / (2 * HD), "pairmin": 1.25}
        rows = []
        for r in heads:
            li, h = r["layer"], r["head"]
            for ti, tl in enumerate(PROBE_TILINGS):
                sel = (li, h, ti, 4)
                keys = st["keys"][sel]
                tiles = st["tiles_elig"][sel]
                row = {"layer": li, "head": h, "tiling": tl, "oracle_rounding": float(st["oracle_keys"][sel][0] / keys),
                       "tiles_gap_ge_15": float(st["gap_hist"][sel][4:].sum() / max(tiles, 1)), "bounds": {}}
                for b, mc in meta.items():
                    bi = M.BOUNDS.index(b)
                    above = float(st["loose"][sel][bi] / max(tiles, 1))
                    best = None
                    for vi, vn in enumerate(M.VARIANTS):
                        sk = st["cert_keys"][sel][bi, vi]
                        if sk < 0:
                            continue
                        extra = (mc + (1.0 if vn.startswith("greedy_tile") else 0.0)) * tiles
                        sav = 1 - (keys - sk + extra) / keys
                        if best is None or sav > best["saving"]:
                            best = {"certificate": vn, "skip": float(sk / keys), "saving": float(sav)}
                    row["bounds"][b] = {"above_tile_max_nats": above, **best}
                rows.append(row)
        res[name] = rows
        for row in rows:
            bb = max(row["bounds"], key=lambda b: row["bounds"][b]["skip"])
            x = row["bounds"][bb]
            print(f"{name} {row['layer']}.{row['head']} {row['tiling']}: oracle {row['oracle_rounding']:.3f}, "
                  f"best {bb} skip {x['skip']:.3f} saving {x['saving']:+.3f} ({x['certificate']}), "
                  f"{bb} above max {x['above_tile_max_nats']:.1f} nats", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
