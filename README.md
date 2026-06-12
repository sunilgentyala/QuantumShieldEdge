# QuantumShieldEdge

**Experimental artifacts for the paper:**

> *QuantumShield-Edge: A Hybrid QKD-Integrated Federated Learning Framework for Secure Consumer IoT Intelligence at the Network Edge*

IEEE CCNC 2027 submission | Sunil Gentyala (HCLTech, IEEE Senior Member #101760715)

---

## Overview

QuantumShieldEdge couples **quantum key distribution (QKD)**-secured gradient transmission with a **variational quantum circuit (VQC)**-based Byzantine anomaly detector for consumer IoT federated learning at the network edge.

Key results (50-node CIFAR-10 testbed, 100 FL rounds):

| Configuration | No Attack | 20% Byzantine |
|---|---|---|
| Vanilla FedAvg | 89.9% | 71.4% |
| Classical Krum | 89.1% | 81.7% |
| **QuantumShieldEdge** | **88.6%** | **85.2%** |

- VQAD detection: **94.7% TPR, 3.2% FPR**
- Key consumption: **22.3 kbps** average (within 40 kbps metropolitan QKD rate)
- OTP encryption latency: **4.2 ms/round** at the EAN

---

## Repository Structure

```
src/
  qkl/          QKD Key Management Layer (BB84 emulator, key buffer)
  qsal/         Quantum-Secured Aggregation Layer (OTP encryption, Top-K sparsification)
  vqad/         Variational Quantum Anomaly Detector (4-qubit VQC, fingerprint extractor)
  fl/           Federated Learning orchestration (Flower server + client, CNN model)
  utils/        Byzantine attack implementations (min-max, label-flip, sign-flip)
experiments/
  config/       YAML configuration files
  run_simulation.py   Main simulation runner
  train_vqad.py       Offline VQAD training
results/
  table1_model_accuracy.csv     Table I data (Section VI-B)
  vqad_detection_rates.csv      VQAD performance metrics (Section VI-C)
  key_consumption_stats.csv     Key rate and latency data (Sections V-B, VI-D/E)
evidence/
  logs/         Sample experiment output logs
models/
  vqad_weights.npy    Pre-trained VQAD circuit parameters
```

---

## Requirements

```
Python 3.11
pennylane==0.36.0
flwr==1.8.0
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.26.0
scipy>=1.11.0
pyyaml>=6.0.1
```

Install:
```bash
pip install -r requirements.txt
```

---

## Quick Start

**Train the VQAD offline:**
```bash
python experiments/train_vqad.py --epochs 100 --seed 42
```

**Run the full simulation (key rate + VQAD evaluation):**
```bash
python experiments/run_simulation.py --config experiments/config/default.yaml
```

**Run a specific scenario:**
```bash
python experiments/run_simulation.py --scenario key_rate
python experiments/run_simulation.py --scenario vqad
```

Results are written to `results/simulation_results.json` and `evidence/logs/simulation_run.log`.

---

## Architecture

### Three-Tier Network (Section III-A)
- **IoT leaf devices** train local models on private data
- **Edge Aggregation Nodes (EANs)** perform first-level aggregation within geographic clusters
- **Cloud aggregator** fuses cluster-level results into a global model

### QKD Key Management Layer (Section IV-B)
Runs decoy-state BB84. Each EAN maintains a key buffer pre-loaded with 3 rounds of material. Proactive re-keying triggers at 40% buffer occupancy, decoupling QKD timing from FL round timing.

### Quantum-Secured Aggregation Layer (Section IV-C)
Top-K gradient sparsification at 5% density reduces key demand by 20x. OTP encryption: `c = g XOR k` where `k = REQUEST_KEY(|g|)`.

### Variational Quantum Anomaly Detector (Section IV-D)
Four-dimensional gradient fingerprint `[L2_norm, cosine_sim, excess_kurtosis, pos_ratio]` is angle-encoded into a 4-qubit register. Three alternating layers of Ry/Rz + CNOT brick-wall (48 parameters). Updates with `<Z> < 0` are flagged Byzantine and down-weighted by 0.1.

---

## Citation

If you use this code, please cite:

```
S. Gentyala et al., "QuantumShield-Edge: A Hybrid QKD-Integrated Federated Learning
Framework for Secure Consumer IoT Intelligence at the Network Edge,"
IEEE CCNC 2027.
```

---

## License

MIT License. See [LICENSE](LICENSE).

> **Note:** The manuscript is under IEEE copyright. The paper PDF is not included in this repository.
