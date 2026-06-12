"""
One-time pad encryption for the Quantum-Secured Aggregation Layer.

Implements Equation (2) from Section IV-C:
    c_i^(t) = g_i^(t) XOR k_i^(t)
where k_i^(t) = REQUEST_KEY(|g_i^(t)|).
"""

import numpy as np


class OTPEncryptor:
    """Stateless OTP encrypt/decrypt using key material from a KeyBuffer."""

    def encrypt(self, gradient_bytes: bytes, key_bytes: bytes) -> bytes:
        if len(key_bytes) < len(gradient_bytes):
            raise ValueError(
                f"Key too short: need {len(gradient_bytes)} B, got {len(key_bytes)} B."
            )
        g = np.frombuffer(gradient_bytes, dtype=np.uint8)
        k = np.frombuffer(key_bytes[: len(gradient_bytes)], dtype=np.uint8)
        return (g ^ k).tobytes()

    def decrypt(self, ciphertext_bytes: bytes, key_bytes: bytes) -> bytes:
        # OTP decryption is identical to encryption (XOR is its own inverse)
        return self.encrypt(ciphertext_bytes, key_bytes)

    @staticmethod
    def gradients_to_bytes(indices: np.ndarray, values: np.ndarray) -> bytes:
        """Pack sparsified (indices, values) pair into a byte string."""
        idx_bytes = indices.astype(np.int32).tobytes()
        val_bytes = values.astype(np.float32).tobytes()
        header = np.array([len(idx_bytes), len(val_bytes)], dtype=np.int64).tobytes()
        return header + idx_bytes + val_bytes

    @staticmethod
    def bytes_to_gradients(data: bytes):
        """Unpack byte string back to (indices, values)."""
        header = np.frombuffer(data[:16], dtype=np.int64)
        idx_len, val_len = int(header[0]), int(header[1])
        offset = 16
        indices = np.frombuffer(data[offset: offset + idx_len], dtype=np.int32)
        offset += idx_len
        values = np.frombuffer(data[offset: offset + val_len], dtype=np.float32)
        return indices, values
