"""
Byzantine attack implementations for simulation (Section VI).

Three attack variants evaluated in the paper:
  - min_max    : min-max gradient poisoning (scales update to maximize harm)
  - label_flip : simulated label-flip poisoning (reverses gradient direction)
  - sign_flip  : negates all gradient components
"""

import numpy as np


class ByzantineAttack:
    """Static factory for applying Byzantine gradient manipulations."""

    @staticmethod
    def apply(
        gradients: list[np.ndarray],
        attack_type: str = "min_max",
        scale: float = 10.0,
    ) -> list[np.ndarray]:
        """
        Apply the specified attack to a list of gradient arrays.

        Parameters
        ----------
        gradients   : list of gradient arrays (one per model layer)
        attack_type : "min_max" | "label_flip" | "sign_flip"
        scale       : amplification factor for min_max attack
        """
        if attack_type == "min_max":
            return ByzantineAttack._min_max(gradients, scale)
        elif attack_type == "label_flip":
            return ByzantineAttack._label_flip(gradients)
        elif attack_type == "sign_flip":
            return ByzantineAttack._sign_flip(gradients)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    @staticmethod
    def _min_max(gradients: list[np.ndarray], scale: float) -> list[np.ndarray]:
        """Scale gradient to maximise deviation from the honest average."""
        return [g * scale for g in gradients]

    @staticmethod
    def _label_flip(gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Simulate label-flip effect by reversing gradient sign."""
        return [-g for g in gradients]

    @staticmethod
    def _sign_flip(gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Negate all gradient components."""
        return [-g for g in gradients]
