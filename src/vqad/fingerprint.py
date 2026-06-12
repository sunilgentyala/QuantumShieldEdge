"""
Four-dimensional gradient fingerprint extractor for the VQAD.

Encodes gradient anomaly signals without exposing raw update data,
preserving participant privacy as described in Section IV-D.

Features:
  f0 — L2 norm of the gradient update
  f1 — cosine similarity to the current global model
  f2 — element-wise excess kurtosis
  f3 — ratio of positive to total non-zero components
"""

import numpy as np
from scipy.stats import kurtosis


class GradientFingerprint:
    """Extract the 4-dimensional anomaly fingerprint from a gradient update."""

    def extract(
        self, gradient: np.ndarray, global_model: np.ndarray
    ) -> np.ndarray:
        """
        Return a 4-element float32 array: [l2_norm, cos_sim, excess_kurtosis, pos_ratio].

        Values are clipped to [-1, 1] to match the Ry angle-encoding range
        expected by the VQC input register.
        """
        flat_g = gradient.flatten().astype(np.float64)
        flat_m = global_model.flatten().astype(np.float64)

        f0 = self._l2_norm(flat_g)
        f1 = self._cosine_similarity(flat_g, flat_m)
        f2 = self._excess_kurtosis(flat_g)
        f3 = self._pos_ratio(flat_g)

        raw = np.array([f0, f1, f2, f3], dtype=np.float32)
        # Normalize to [-1, 1] for Ry angle encoding (multiply by pi externally)
        return np.clip(raw, -1.0, 1.0)

    @staticmethod
    def _l2_norm(v: np.ndarray) -> float:
        norm = np.linalg.norm(v)
        # Normalize against a typical gradient magnitude scale
        return float(np.tanh(norm / (np.std(v) * np.sqrt(len(v)) + 1e-9)))

    @staticmethod
    def _cosine_similarity(g: np.ndarray, m: np.ndarray) -> float:
        dg = np.linalg.norm(g)
        dm = np.linalg.norm(m)
        if dg < 1e-12 or dm < 1e-12:
            return 0.0
        return float(np.dot(g, m) / (dg * dm))

    @staticmethod
    def _excess_kurtosis(v: np.ndarray) -> float:
        if len(v) < 4:
            return 0.0
        k = float(kurtosis(v, fisher=True, bias=False))
        # Tanh-compress: benign gradients typically have kurtosis near 0
        return float(np.tanh(k / 3.0))

    @staticmethod
    def _pos_ratio(v: np.ndarray) -> float:
        nonzero = v[v != 0.0]
        if len(nonzero) == 0:
            return 0.0
        pos_count = np.sum(nonzero > 0.0)
        ratio = pos_count / len(nonzero)
        # Map [0,1] -> [-1, 1]
        return float(2.0 * ratio - 1.0)
