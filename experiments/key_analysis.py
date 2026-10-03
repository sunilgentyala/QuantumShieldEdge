"""Analytical and Monte Carlo QKD-side results (no FL training needed).

 A. Key demand per EAN link per round vs model size and density (exact wire-format bit counts).
 B. Secret-key fraction vs QBER and finite block length; achievable secret rate for the nominal link.
 C. BB84 Monte Carlo: eavesdropper detection (intercept-resend fraction vs estimated QBER).
 D. Shared QKD link contention: N EANs on one link vs one link each; outage probability vs provisioning.
 E. Trusted-relay chain loading: required per-hop rate for a star (shared backbone hop) topology.
Output: results/key_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT  # noqa: E402

from qse.crypto import otp_wc as ow  # noqa: E402
from qse.model import SmallCNN, n_params  # noqa: E402
from qse.qkd.bb84 import run_bb84  # noqa: E402
from qse.qkd.finite_key import asymptotic_fraction, asymptotic_threshold, finite_fraction  # noqa: E402
from qse.qkd.keypool import LinkConfig, QKDLink  # noqa: E402

TR = 60.0


def demand_bits(P: int, dens_up: float, dens_down: float, mode: str) -> int:
    ku, kd = max(1, int(dens_up * P)), max(1, int(dens_down * P))
    up = ow.message_key_bits(ow.sparse_payload_bits(ku, P))
    if mode == "sparse_otp":
        down = ow.message_key_bits(ow.sparse_payload_bits(kd, P))
    elif mode == "dense_otp":
        down = ow.message_key_bits(ow.dense_payload_bits(P))
    else:
        down = 8 * ow.TAG_BYTES
    return up + down


def main():
    out = {}
    P_sim = n_params(SmallCNN((16, 32, 64)))
    out["P_sim"] = P_sim

    # A. key demand per link per round and required rate
    models = {"sim CNN (24k)": P_sim, "100k": 100_000, "500k": 500_000, "1M": 1_000_000, "5M": 5_000_000}
    dens = [0.01, 0.02, 0.05, 0.10, 0.25]
    A = {}
    for name, P in models.items():
        A[name] = {"P": P}
        for mode in ("sparse_otp", "dense_otp", "mac_only"):
            A[name][mode] = {f"{d}": demand_bits(P, d, d, mode) / TR / 1000.0 for d in dens}
    out["A_required_kbps_per_link"] = A
    out["A_wire_note"] = "kbps = key bits per round (up + down, incl. headers and 64-bit MAC pads) / 60 s / 1000"

    # B. finite-key fractions and achievable rate (sifted 60 kbps)
    qs = np.linspace(0.0, 0.12, 121)
    B = {"qber": qs.tolist(), "asymptotic": asymptotic_fraction(qs).tolist()}
    for n in (10_000, 100_000, 1_000_000, 10_000_000):
        B[f"finite_{n}"] = finite_fraction(qs, n).tolist()
    B["threshold_f1.0"] = asymptotic_threshold(1.0)
    B["threshold_f1.16"] = asymptotic_threshold(1.16)
    out["B_secret_fraction"] = B

    # C. BB84 eavesdropper detection
    rng = np.random.default_rng(0)
    C = []
    for eve in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        qe = [run_bb84(200_000, 0.02, rng, eve).qber_est for _ in range(20)]
        ab = [run_bb84(200_000, 0.02, rng, eve).aborted for _ in range(20)]
        C.append(dict(eve_fraction=eve, qber_mean=float(np.mean(qe)), qber_std=float(np.std(qe)),
                      abort_rate=float(np.mean(ab)), theory=0.02 + 0.25 * eve * (1 - 2 * 0.02)))
    out["C_eavesdropper"] = C

    # D. shared-link contention. Demand D bits/round/EAN (sparse_otp, 5%, sim CNN).
    Dbits = demand_bits(P_sim, 0.05, 0.05, "sparse_otp")
    N = 4
    D = {"demand_bits_per_ean_round": Dbits, "N": N}
    for prov in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        res = {}
        for mode in ("one_link_each", "shared_equal_split", "shared_pooled"):
            outage = []
            for seed in range(20):
                # secret rate per link such that the total key made per round = prov * N * D on average
                nominal = LinkConfig()
                probe = QKDLink(nominal, 10 ** 9, seed=seed)
                mean_rate = np.mean([probe.sample_state()[1] for _ in range(400)])
                n_links = N if mode == "one_link_each" else 1
                scale = prov * N * Dbits / TR / (n_links * mean_rate)
                cfg = LinkConfig(rate_scale=scale)
                links = [QKDLink(cfg, 4 * N * Dbits, seed=seed * 10 + i) for i in range(n_links)]
                pools = [0.0] * n_links
                short = 0
                tot = 0
                for r in range(200):
                    gen = []
                    for i, l in enumerate(links):
                        _, rate = l.sample_state()
                        gen.append(rate * TR)
                    if mode == "one_link_each":
                        for i in range(N):
                            pools[i] = min(pools[i] + gen[i], 4 * Dbits)
                            tot += 1
                            if pools[i] >= Dbits:
                                pools[i] -= Dbits
                            else:
                                short += 1
                    else:
                        pools[0] = min(pools[0] + gen[0], 4 * N * Dbits)
                        for i in range(N):
                            tot += 1
                            if mode == "shared_equal_split":
                                # each EAN may only use its 1/N share of the link's key this round
                                share = gen[0] / N
                                if share >= Dbits:
                                    pools[0] -= min(pools[0], Dbits)
                                else:
                                    short += 1
                            else:  # pooled: first come first served from the shared buffer
                                if pools[0] >= Dbits:
                                    pools[0] -= Dbits
                                else:
                                    short += 1
                outage.append(short / tot)
            res[mode] = dict(mean=float(np.mean(outage)), std=float(np.std(outage)))
        D[f"prov_{prov}"] = res
    out["D_shared_link"] = D

    # E. trusted-relay star: k EANs aggregate through one backbone hop to the cloud
    E = {}
    for k in (1, 2, 4, 8, 16):
        per_ean = Dbits / TR / 1000.0
        E[str(k)] = dict(edge_hop_kbps=per_ean, backbone_hop_kbps=k * per_ean)
    out["E_relay_star_required_kbps"] = E

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "key_analysis.json").write_text(json.dumps(out, indent=1))
    print("demand per EAN link per round (sim CNN, 5% up/down):", Dbits, "bits ->", Dbits / TR / 1000, "kbps")
    for k, v in A.items():
        print(k, {m: round(v[m]["0.05"], 2) for m in ("sparse_otp", "dense_otp", "mac_only")})


if __name__ == "__main__":
    main()
