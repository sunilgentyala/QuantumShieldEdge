"""Three-tier QKD-keyed federated learning simulator (leaf -> EAN -> cloud).

Data path per round
  leaf   : local SGD from its EAN's model replica; Byzantine leaves replace/poison their update.
           Leaf->EAN traffic is dense float32 and assumed protected by a computationally secure AEAD whose key is
           established with a post-quantum KEM. It is NOT simulated and consumes no QKD key.
  EAN    : robust / detector-weighted aggregation of its cluster's updates, Top-K sparsification with error
           feedback, fp16 quantisation, wire-format packing, OTP encryption + Wegman-Carter tag with key bits
           taken from the EAN<->cloud QKD link pool.
  cloud  : verifies the tag, decrypts, unpacks, and aggregates ONLY what it decoded. It then sends each EAN a
           sealed (sparse or dense) model delta, which the EAN applies to its replica.

If key is insufficient, a policy decides: 'adapt' (shrink densities), 'wait' (extend the round until key
accrues), 'defer' (skip this round's exchange; error feedback keeps the update), 'pq' (send under computational
security only; counted as a non-information-theoretic round).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

from . import attacks as atk
from .crypto import otp_wc as ow
from .data import assign_clusters, dirichlet_partition, load_cifar10
from .defenses import AGGREGATORS
from .model import SmallCNN, evaluate, get_flat, n_params, set_flat
from .qkd.keypool import LinkConfig, QKDLink
from .vqad.features import fingerprints

_DATA = {}


def get_data():
    if "d" not in _DATA:
        _DATA["d"] = load_cifar10()
    return _DATA["d"]


@dataclass
class SimConfig:
    seed: int = 0
    n_clients: int = 50
    cluster_sizes: tuple = (13, 12, 13, 12)
    rounds: int = 60
    alpha: float = 0.5
    local_epochs: int = 1
    local_steps: int = 5             # if >0: fixed number of minibatch steps per round (overrides epochs)
    widths: tuple = (16, 32, 64)
    batch_size: int = 32
    lr: float = 0.05
    byz_frac: float = 0.2
    attack: str = "none"             # none | sign_flip | scale | label_flip | alie | min_max
    aggregator: str = "fedavg"       # see defenses.AGGREGATORS
    detector_path: str | None = None  # npz with detector weights for vqad/mlp/linear
    flagged_weight: float = 0.1
    pipeline: str = "qkd"            # dense (ideal channel, no compression) | qkd
    density_up: float = 0.05
    density_down: float = 0.05
    down_mode: str = "sparse_otp"    # sparse_otp | dense_otp | mac_only
    policy: str = "adapt"            # adapt | wait | defer | pq
    round_seconds: float = 60.0
    prefill_rounds: int = 3
    capacity_bits: int = 4_000_000
    link: dict = field(default_factory=dict)  # LinkConfig overrides
    eval_every: int = 5
    min_density: float = 0.005
    wait_cap_seconds: float = 600.0
    record_fingerprints: bool = False
    qber_events: tuple = ()          # ((first_round, last_round, qber), ...) transient QBER overrides


def _train_leaf(model, flat0, x, y, cfg, gen, flip_labels):
    set_flat(model, flat0)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    n = len(x)
    steps = cfg.local_steps if cfg.local_steps > 0 else cfg.local_epochs * math.ceil(n / cfg.batch_size)
    perm = torch.randperm(n, generator=gen)
    pos = 0
    for _ in range(steps):
        if pos + cfg.batch_size > n:
            perm = torch.randperm(n, generator=gen)
            pos = 0
        b = perm[pos:pos + cfg.batch_size]
        pos += cfg.batch_size
        if len(b) < 2:
            continue
        yb = (y[b] + 1) % 10 if flip_labels else y[b]
        opt.zero_grad()
        F.cross_entropy(model(x[b]), yb).backward()
        opt.step()
    return get_flat(model) - flat0


class Simulation:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.xtr, self.ytr, self.xte, self.yte = get_data()
        self.parts = dirichlet_partition(self.ytr.numpy(), cfg.n_clients, cfg.alpha, self.rng)
        self.clusters = assign_clusters(cfg.n_clients, list(cfg.cluster_sizes), self.rng)
        self.byz = np.zeros(cfg.n_clients, dtype=bool)
        if cfg.attack != "none":
            for c in self.clusters:
                k = int(round(cfg.byz_frac * len(c)))
                self.byz[self.rng.choice(c, k, replace=False)] = True
        self.model = SmallCNN(cfg.widths)
        self.P = n_params(self.model)
        self.w_g = get_flat(self.model)
        self.n_ean = len(self.clusters)
        self.w_hat = [self.w_g.clone() for _ in range(self.n_ean)]
        self.resid = [torch.zeros(self.P) for _ in range(self.n_ean)]
        self.detector = self._load_detector()
        self.links = None
        self.mac_keys = None
        if cfg.pipeline == "qkd":
            lc = LinkConfig(**cfg.link)
            self.links = [QKDLink(lc, cfg.capacity_bits, seed=cfg.seed * 100 + i) for i in range(self.n_ean)]
            for l in self.links:
                for _ in range(cfg.prefill_rounds):
                    l.advance(cfg.round_seconds)
            # Wegman-Carter hash keys come from a one-time trusted setup (not from the QKD pool);
            # only the per-message tag pads are consumed from the pool.
            krng = np.random.default_rng(cfg.seed + 31337)
            self.mac_keys = [int(krng.integers(1, ow.P61)) for _ in self.links]
        self.log = dict(acc=[], acc_round=[], tp=[], fp=[], fn=[], tn=[], key_up_bits=[], key_down_bits=[],
                        pool_level=[], q_rate_bps=[], qber=[], n_adapted=[], n_deferred=[], n_pq=[], n_ok=[],
                        delay_s=[], density_up_eff=[], auth_fail=0, macs_verified=0)
        self.fp_log: list = []

    def _load_detector(self):
        c = self.cfg
        if c.aggregator not in ("vqad", "mlp", "linear"):
            return None
        from .vqad.detector import VQAD, MLPDetector, LogisticDetector
        det = {"vqad": VQAD, "mlp": MLPDetector, "linear": LogisticDetector}[c.aggregator]()
        z = np.load(c.detector_path)
        with torch.no_grad():
            for p, name in zip(det._params(), sorted(z.files)):
                p.copy_(torch.tensor(z[name]))
        return det

    # ------------------------------------------------------------------ key handling
    def _need(self, du, dd, k_up, k_down):
        c = self.cfg
        up = ow.message_key_bits(ow.sparse_payload_bits(k_up, self.P))
        if c.down_mode == "sparse_otp":
            down = ow.message_key_bits(ow.sparse_payload_bits(k_down, self.P))
        elif c.down_mode == "dense_otp":
            down = ow.message_key_bits(ow.dense_payload_bits(self.P))
        else:  # mac_only: tag pad only
            down = 8 * ow.TAG_BYTES
        return up, down

    def _plan(self, i):
        """Decide how EAN i's exchange proceeds this round. Returns (mode, k_up, k_down)."""
        c = self.cfg
        link = self.links[i]
        ku, kd = max(1, int(c.density_up * self.P)), max(1, int(c.density_down * self.P))
        delay = 0.0
        while True:
            up, down = self._need(c.density_up, c.density_down, ku, kd)
            lvl = link.ean_pool.level_bits()
            if up + down <= lvl:
                return "ok", ku, kd, delay
            if c.policy == "pq":
                return "pq", ku, kd, delay
            if c.policy == "wait" and delay < c.wait_cap_seconds:
                link.advance(1.0)
                delay += 1.0
                continue
            if c.policy == "adapt" and c.down_mode == "sparse_otp":
                fixed = 2 * 8 * (ow.HEADER_BYTES + ow.TAG_BYTES)
                per_entry = ow.index_bits(self.P) + 16
                budget = lvl - fixed
                k_each = int(budget // (2 * per_entry)) if budget > 0 else 0
                k_each = min(k_each, ku)
                # the closed form ignores byte rounding of the packed indices; verify the exact cost
                while k_each > 0 and sum(self._need(c.density_up, c.density_down, k_each, k_each)) > lvl:
                    k_each -= 1
                if k_each >= max(1, int(c.min_density * self.P)):
                    return "adapted", k_each, k_each, delay
            return "defer", 0, 0, delay

    def _send(self, i, payload: bytes) -> bytes:
        link = self.links[i]
        kp, tp = link.ean_pool.take(len(payload)), link.ean_pool.take(ow.TAG_BYTES)
        kp2, tp2 = link.cloud_pool.take(len(payload)), link.cloud_pool.take(ow.TAG_BYTES)
        assert kp == kp2 and tp == tp2 and kp is not None, "pool desynchronised"
        msg = ow.seal(payload, kp, tp, self.mac_keys[i])
        return msg, kp, tp

    # ------------------------------------------------------------------ one round
    def run_round(self, r: int) -> None:
        c = self.cfg
        P = self.P
        n = c.n_clients
        # advance QKD links by one round of wall-clock
        qs, rates = [], []
        if self.links:
            for l in self.links:
                base = LinkConfig(**c.link).qber_override
                l.cfg.qber_override = base
                for r0, r1, qv in c.qber_events:
                    if r0 <= r <= r1:
                        l.cfg.qber_override = qv
                l.advance(c.round_seconds)
                qs.append(l.last_qber)
                rates.append(l.last_rate_bps)

        # ---- leaf training
        U = torch.zeros(n, P)
        for i, cl in enumerate(self.clusters):
            for j in cl:
                idx = torch.from_numpy(self.parts[j])
                flip = bool(self.byz[j] and c.attack == "label_flip")
                U[j] = _train_leaf(self.model, self.w_hat[i], self.xtr[idx], self.ytr[idx], c, self.gen, flip)
        # ---- model poisoning (omniscient adversary sees honest updates)
        if c.attack in ("sign_flip", "scale", "alie", "min_max"):
            honest = U[~torch.from_numpy(self.byz)]
            nb = int(self.byz.sum())
            if c.attack in ("alie", "min_max"):
                mal = atk.craft(c.attack, honest, None, n, nb)
                for j in np.where(self.byz)[0]:
                    U[j] = mal
            else:
                for j in np.where(self.byz)[0]:
                    U[j] = atk.craft(c.attack, honest, U[j], n, nb)

        # ---- EAN aggregation
        ns = torch.tensor([len(p) for p in self.parts], dtype=torch.float64)
        tp = fp = fn = tn = 0
        deliveries = []  # (weight, decoded vector)
        key_up = key_down = 0
        n_adapt = n_def = n_pq = n_ok = 0
        delay_max = 0.0
        dens_eff = []
        for i, cl in enumerate(self.clusters):
            Uc = U[cl]
            f_assumed = max(1, int(math.ceil(c.byz_frac * len(cl))))
            agg, wt, flags = AGGREGATORS[c.aggregator](
                Uc, ns[cl], f=f_assumed, detector=self.detector, flagged_weight=c.flagged_weight)
            truth = self.byz[cl]
            if c.record_fingerprints:
                self.fp_log.append((r, i, fingerprints(Uc).numpy(), truth.copy()))
            if flags is not None:
                fl = np.asarray(flags, dtype=bool)
                tp += int((fl & truth).sum()); fp += int((fl & ~truth).sum())
                fn += int((~fl & truth).sum()); tn += int((~fl & ~truth).sum())

            if c.pipeline == "dense":
                deliveries.append((i, wt, agg.float(), None))
                continue

            v = agg.float() + self.resid[i]
            mode, ku, kd, delay = self._plan(i)
            delay_max = max(delay_max, delay)
            if mode == "defer":
                self.resid[i] = v
                n_def += 1
                deliveries.append((i, 0.0, None, mode))
                continue
            top = v.abs().topk(ku).indices.sort().values
            vals = v[top].half()
            dec_vec = torch.zeros(P)
            dec_vec[top] = vals.float()
            self.resid[i] = v - dec_vec
            payload = ow.pack_sparse(r, top.numpy(), vals.numpy(), P)
            if mode == "pq":
                n_pq += 1
                got = ow.unpack_sparse(payload, P)
            else:
                msg, kp, tpd = self._send(i, payload)
                key_up += 8 * (len(payload) + ow.TAG_BYTES)
                # --- cloud side: verify, decrypt (fresh pool pads identical to sender's by construction)
                try:
                    clear = ow.open_sealed(msg, kp, tpd, self.mac_keys[i])
                    self.log["macs_verified"] += 1
                except ow.AuthError:
                    self.log["auth_fail"] += 1
                    continue
                got = ow.unpack_sparse(clear, P)
                n_adapt += int(mode == "adapted")
                n_ok += int(mode == "ok")
            _, gi, gv = got
            dec = torch.zeros(P)
            dec[torch.from_numpy(gi)] = torch.from_numpy(gv)
            deliveries.append((i, wt, dec, (mode, kd)))
            dens_eff.append(ku / P)

        # ---- cloud aggregation
        tot = sum(w for _, w, d, _ in deliveries if d is not None)
        if tot > 0:
            step = sum((w / tot) * d for _, w, d, _ in deliveries if d is not None)
            self.w_g = self.w_g + step

        # ---- downstream
        if c.pipeline == "dense":
            self.w_hat = [self.w_g.clone() for _ in range(self.n_ean)]
        else:
            for i, w, d, meta in deliveries:
                if meta is None or d is None:
                    continue
                mode, kd = meta
                delta = self.w_g - self.w_hat[i]
                if c.down_mode == "sparse_otp":
                    kd_eff = kd
                    top = delta.abs().topk(kd_eff).indices.sort().values
                    payload = ow.pack_sparse(r, top.numpy(), delta[top].half().numpy(), P)
                    if mode == "pq":
                        _, gi, gv = ow.unpack_sparse(payload, P)
                    else:
                        msg, kp, tpd = self._send(i, payload)
                        key_down += 8 * (len(payload) + ow.TAG_BYTES)
                        _, gi, gv = ow.unpack_sparse(ow.open_sealed(msg, kp, tpd, self.mac_keys[i]), P)
                        self.log["macs_verified"] += 1
                    upd = torch.zeros(P)
                    upd[torch.from_numpy(gi)] = torch.from_numpy(gv)
                else:
                    payload = ow.pack_dense(r, delta.numpy())
                    if mode == "pq":
                        _, gv = ow.unpack_dense(payload)
                    elif c.down_mode == "dense_otp":
                        msg, kp, tpd = self._send(i, payload)
                        key_down += 8 * (len(payload) + ow.TAG_BYTES)
                        _, gv = ow.unpack_dense(ow.open_sealed(msg, kp, tpd, self.mac_keys[i]))
                        self.log["macs_verified"] += 1
                    else:  # mac_only: authenticated, NOT confidential
                        link = self.links[i]
                        pad_s, pad_r = link.ean_pool.take(ow.TAG_BYTES), link.cloud_pool.take(ow.TAG_BYTES)
                        tag = ow.poly_hash(payload, self.mac_keys[i]) ^ int.from_bytes(pad_s, "little")
                        if ow.poly_hash(payload, self.mac_keys[i]) ^ int.from_bytes(pad_r, "little") != tag:
                            self.log["auth_fail"] += 1
                        else:
                            self.log["macs_verified"] += 1
                        key_down += 8 * ow.TAG_BYTES
                        _, gv = ow.unpack_dense(payload)
                    upd = torch.from_numpy(gv.copy())
                self.w_hat[i] = self.w_hat[i] + upd

        L = self.log
        L["tp"].append(tp); L["fp"].append(fp); L["fn"].append(fn); L["tn"].append(tn)
        L["key_up_bits"].append(key_up); L["key_down_bits"].append(key_down)
        L["n_adapted"].append(n_adapt); L["n_deferred"].append(n_def); L["n_pq"].append(n_pq); L["n_ok"].append(n_ok)
        L["delay_s"].append(delay_max)
        L["density_up_eff"].append(float(np.mean(dens_eff)) if dens_eff else 0.0)
        if self.links:
            L["pool_level"].append([l.ean_pool.level() for l in self.links])
            L["q_rate_bps"].append(rates); L["qber"].append(qs)
        if r % c.eval_every == 0 or r == c.rounds:
            set_flat(self.model, self.w_g)
            L["acc"].append(evaluate(self.model, self.xte, self.yte)); L["acc_round"].append(r)

    def run(self) -> dict:
        t0 = time.time()
        for r in range(1, self.cfg.rounds + 1):
            self.run_round(r)
        out = dict(cfg=asdict(self.cfg), n_params=self.P, log=self.log, seconds=time.time() - t0,
                   final_acc=self.log["acc"][-1])
        return out
