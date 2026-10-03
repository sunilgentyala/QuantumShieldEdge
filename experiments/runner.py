"""Resumable multiprocess experiment runner. One JSON per run in results/raw/<suite>/<name>.json."""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _work(args):
    name, cfgd, outdir = args
    import torch
    torch.set_num_threads(1)
    from qse.sim import SimConfig, Simulation
    out = Path(outdir) / f"{name}.json"
    if out.exists():
        return name, "cached"
    cfg = SimConfig(**{**cfgd, "cluster_sizes": tuple(cfgd.get("cluster_sizes", (13, 12, 13, 12))),
                       "widths": tuple(cfgd.get("widths", (16, 32, 64))),
                       "qber_events": tuple(tuple(e) for e in cfgd.get("qber_events", ()))})
    sim = Simulation(cfg)
    res = sim.run()
    if cfg.record_fingerprints:
        res["fingerprints"] = [dict(round=r, ean=i, X=X.tolist(), y=y.tolist()) for r, i, X, y in sim.fp_log]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    return name, f"{res['seconds']:.0f}s acc={res['final_acc']:.3f}"


def run_all(jobs: list[tuple[str, dict]], suite: str, workers: int = 8) -> None:
    outdir = ROOT / "results" / "raw" / suite
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [(n, c, str(outdir)) for n, c in jobs]
    t0 = time.time()
    done = 0
    with Pool(workers) as p:
        for name, status in p.imap_unordered(_work, todo):
            done += 1
            print(f"[{suite}] {done}/{len(todo)} {name}: {status}  (elapsed {time.time() - t0:.0f}s)", flush=True)


def load_suite(suite: str) -> dict:
    d = ROOT / "results" / "raw" / suite
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
