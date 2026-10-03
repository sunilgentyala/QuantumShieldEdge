import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import run_all

if __name__ == "__main__":
    jobs = []
    for lr, st in [(0.05, 10), (0.1, 5), (0.1, 10), (0.2, 5), (0.2, 10), (0.1, 20)]:
        jobs.append((f"lr{lr}_st{st}", dict(seed=100, pipeline="dense", lr=lr, local_steps=st, rounds=60, eval_every=10)))
    run_all(jobs, "probe", workers=6)
