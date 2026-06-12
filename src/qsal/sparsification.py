"""
Top-K gradient sparsification for the Quantum-Secured Aggregation Layer.

Reduces key demand by factor K/M (default 0.05 = 5% density) as described
in Section IV-C. Only the K largest-magnitude components and their indices
are transmitted, reducing the OTP key consumption proportionally.
"""

import numpy as np


class TopKSparsifier:
    """
    Top-K gradient sparsification.

    Parameters
    ----------
    density : float
        Fraction of gradient components to keep (default 0.05 for 5%).
    """

    def __init__(self, density: float = 0.05) -> None:
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density}")
        self.density = density

    def sparsify(self, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, values) of the top-K magnitude components.

        K = max(1, round(density * len(gradient))).
        """
        flat = gradient.flatten()
        k = max(1, round(self.density * len(flat)))
        top_k_idx = np.argpartition(np.abs(flat), -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(np.abs(flat[top_k_idx]))[::-1]]
        return top_k_idx.astype(np.int32), flat[top_k_idx].astype(np.float32)

    def reconstruct(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        original_shape,
    ) -> np.ndarray:
        """Reconstruct full-size gradient from sparse representation (zeros elsewhere)."""
        flat = np.zeros(int(np.prod(original_shape)), dtype=np.float32)
        flat[indices] = values
        return flat.reshape(original_shape)

    def key_bits_required(self, model_bits: int) -> int:
        """Return the number of key bits required after sparsification."""
        return max(1, round(self.density * model_bits))
