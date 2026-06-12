<div align="center">

# QuantumShield-Edge

### A Hybrid QKD-Integrated Federated Learning Framework for Secure Consumer IoT Intelligence at the Network Edge

[![IEEE CCNC 2027](https://img.shields.io/badge/IEEE%20CCNC-2027-blue?style=flat-square&logo=ieee)](https://github.com/sunilgentyala/QuantumShieldEdge)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PennyLane 0.36](https://img.shields.io/badge/PennyLane-0.36-512BD4?style=flat-square)](https://pennylane.ai/)
[![Flower 1.8](https://img.shields.io/badge/Flower-1.8-FF6B6B?style=flat-square)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Project%20Page-Live-6366f1?style=flat-square&logo=github)](https://sunilgentyala.github.io/QuantumShieldEdge/)

**Sunil Gentyala** &nbsp;|&nbsp; HCLTech &nbsp;|&nbsp; IEEE Senior Member #101760715 &nbsp;|&nbsp; CISM &nbsp;|&nbsp; CCZT  
**Suresh Kumar Darisi** &nbsp;|&nbsp; Rocket Software &nbsp;|&nbsp; IEEE Member #101925007

[Project Page](https://sunilgentyala.github.io/QuantumShieldEdge/) &nbsp;&bull;&nbsp; [Paper (CCNC 2027)](#citation) &nbsp;&bull;&nbsp; [Results](results/) &nbsp;&bull;&nbsp; [Evidence](evidence/)

</div>

---

## Overview

Consumer IoT federated learning depends on edge aggregation channels protected only by classical ciphers that a quantum-capable adversary can break. **QuantumShield-Edge** closes this gap by embedding **quantum key distribution (QKD)** directly into the FL gradient pipeline and pairing it with a **variational quantum circuit (VQC)** anomaly detector that flags Byzantine participants without ever exposing raw gradient data.

The result: information-theoretic gradient channel security plus 94.7% Byzantine detection accuracy, both feasible on current metropolitan QKD infrastructure.

---

## Key Results

Evaluated on a 50-node heterogeneous IoT testbed (CIFAR-10, Dirichlet α=0.5, 100 FL rounds, averaged over 5 seeds):

| Configuration | No Attack | 20% Byzantine (min-max) |
|:---|:---:|:---:|
| Vanilla FedAvg | 89.9% | 71.4% |
| Classical Krum | 89.1% | 81.7% |
| **QuantumShield-Edge (ours)** | **88.6%** | **85.2%** |

| Metric | Value |
|:---|:---|
| VQAD True Positive Rate | **94.7%** (over all 100 rounds, all attack types) |
| VQAD False Positive Rate | **3.2%** (< 1% after round 20) |
| Mean key consumption | **22.3 kbps** (within 40 kbps metropolitan QKD rate) |
| OTP encryption latency | **4.2 ms / round** at the EAN |
| VQAD inference latency | **38 ms / round** (CPU); negligible on QPU |
| Minimum buffer occupancy | **18%** under nominal QBER conditions |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLOUD AGGREGATOR                          │
│                    (Global FedAvg + VQAD)                       │
└───────────────┬─────────────┬──────────────┬────────────────────┘
                │  QKD-OTP    │  QKD-OTP     │  QKD-OTP
                │  channel    │  channel     │  channel
    ┌───────────▼──┐  ┌───────▼──┐  ┌───────▼──┐  ┌────────────┐
    │    EAN-1     │  │  EAN-2   │  │  EAN-3   │  │   EAN-4    │
    │  (13 nodes)  │  │ (12 nodes│  │ (13 nodes│  │ (12 nodes) │
    │  QKL buffer  │  │ QKL buf. │  │ QKL buf. │  │ QKL buf.   │
    │  Top-K QSAL  │  │ Top-K    │  │ Top-K    │  │ Top-K      │
    └──┬──┬──┬─────┘  └──────────┘  └──────────┘  └────────────┘
       │  │  │
  IoT leaf devices (sensors · wearables · smart appliances)
  Local training on private data — post-quantum leaf-to-EAN links

Legend:  QKL = QKD Key Management Layer (decoy-state BB84, 40-kbps metropolitan fiber)
         QSAL = Quantum-Secured Aggregation Layer (Top-K sparsification + OTP c=g⊕k)
         VQAD = Variational Quantum Anomaly Detector (4-qubit VQC, 48 parameters)
```

### Three-Layer Pipeline

**QKD Key Management Layer (Section IV-B)**
Runs decoy-state BB84. Each EAN pre-loads 3 rounds of keying material. Proactive re-keying fires when buffer occupancy drops below 40%, fully decoupling QKD timing from FL round timing.

```python
key = buffer.request_key(n_bits)    # atomic dequeue, never reused
level = buffer.key_level()          # drives adaptive sparsification
```

**Quantum-Secured Aggregation Layer (Section IV-C)**
Top-K sparsification at 5% density cuts key demand 20-fold. OTP encryption is unconditionally secure: `c = g ⊕ k` where `k = REQUEST_KEY(|g|)`. No computational hardness assumption at any layer.

```python
idx, vals = sparsifier.sparsify(gradient)   # K/M = 0.05
ciphertext = encryptor.encrypt(gradient_bytes, key)
```

**Variational Quantum Anomaly Detector (Section IV-D)**
Four gradient statistics are angle-encoded into a 4-qubit register via Ry rotations. Three layers of Ry/Rz + CNOT brick-wall entanglement (48 trainable parameters). `⟨Z⟩ < 0` flags a Byzantine participant, which is down-weighted by 0.1 in aggregation.

```
Features → [L2_norm | cosine_sim | excess_kurtosis | pos_ratio]
                ↓ Ry(π·f) angle encoding on 4 qubits
              [Layer 1: Ry Rz │ CNOT pattern]
              [Layer 2: Ry Rz │ CNOT pattern]
              [Layer 3: Ry Rz │ CNOT pattern]
                ↓ measure ⟨Z⟩ on qubit 0
         ⟨Z⟩ ≥ 0 → benign (weight = 1.0)
         ⟨Z⟩ < 0 → Byzantine flag (weight = 0.1)
```

---

## Repository Structure

```
QuantumShieldEdge/
├── src/
│   ├── qkl/               QKD Key Management Layer
│   │   ├── bb84_emulator.py     Decoy-state BB84 with QBER abort
│   │   └── key_buffer.py        Thread-safe buffer, proactive re-keying
│   ├── qsal/              Quantum-Secured Aggregation Layer
│   │   ├── otp_encryption.py    OTP encrypt/decrypt (Equation 2)
│   │   └── sparsification.py    Top-K gradient sparsification
│   ├── vqad/              Variational Quantum Anomaly Detector
│   │   ├── circuit.py           4-qubit PennyLane VQC
│   │   ├── fingerprint.py       4-dim gradient fingerprint extractor
│   │   └── detector.py          Training (parameter-shift + Adam) + inference
│   ├── fl/                Federated Learning orchestration
│   │   ├── model.py             LightweightCNN (~180K params, CIFAR-10)
│   │   ├── client.py            Flower client with QKL integration
│   │   └── server.py            Flower server with VQAD soft-weighted FedAvg
│   └── utils/
│       └── attacks.py           Byzantine attacks: min-max, label-flip, sign-flip
├── experiments/
│   ├── config/default.yaml      Full experiment configuration
│   ├── run_simulation.py        Reproduces Table I + Sections VI-C/D/E
│   └── train_vqad.py            Offline VQAD training
├── results/
│   ├── table1_model_accuracy.csv
│   ├── vqad_detection_rates.csv
│   └── key_consumption_stats.csv
├── evidence/
│   └── logs/sample_experiment.log
└── models/
    └── vqad_weights.npy         Pre-trained VQAD parameters
```

---

## Installation

```bash
git clone https://github.com/sunilgentyala/QuantumShieldEdge.git
cd QuantumShieldEdge
pip install -r requirements.txt
```

**Requirements:** Python 3.11 · PennyLane 0.36 · Flower (flwr) 1.8 · PyTorch ≥ 2.1 · NumPy ≥ 1.26 · SciPy ≥ 1.11

---

## Usage

**Train the VQAD anomaly detector offline:**
```bash
python experiments/train_vqad.py --epochs 100 --seed 42
# Saves trained weights to models/vqad_weights.npy
```

**Run the full simulation (QKD key rate feasibility + VQAD evaluation):**
```bash
python experiments/run_simulation.py --config experiments/config/default.yaml
```

**Run individual evaluation scenarios:**
```bash
python experiments/run_simulation.py --scenario key_rate   # Section V-B
python experiments/run_simulation.py --scenario vqad       # Section VI-C
python experiments/run_simulation.py --scenario all        # everything
```

Output is written to `results/simulation_results.json` and `evidence/logs/simulation_run.log`.

---

## Key Theoretical Result

**Theorem 1 (Composable Security):** QuantumShield-Edge achieves ε-security against any computationally unbounded adversary on the quantum channel, where ε is bounded by the security parameter of the underlying decoy-state BB84 protocol.

The minimum sustained key rate for a practical deployment:

```
K_min = α × M_sparse × n_c / T_r
      = 1.1 × 400 Kbit × 4 clusters / 60 s
      ≈ 29.3 kbps
```

This falls within the lower bound of current metropolitan QKD hardware (10–100 kbps over 50–80 km fiber), confirming operational feasibility.

---

## Citation

```bibtex
@inproceedings{gentyala2027quantumshield,
  title     = {{QuantumShield-Edge}: A Hybrid {QKD}-Integrated Federated Learning
               Framework for Secure Consumer {IoT} Intelligence at the Network Edge},
  author    = {Gentyala, Sunil and Darisi, Suresh Kumar},
  booktitle = {Proceedings of the IEEE Consumer Communications \& Networking Conference
               (CCNC)},
  year      = {2027},
  publisher = {IEEE}
}
```

---

## License

Released under the [MIT License](LICENSE).

> The manuscript is under IEEE copyright and is not included in this repository.
> All experimental artifacts (source code, configurations, results, pre-trained weights) are freely available here.
