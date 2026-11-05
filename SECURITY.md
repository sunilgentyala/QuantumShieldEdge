# Security policy

## What this project is

QuantumShield-Edge is a **research simulator**. It is not a production cryptographic library and must not be used
to protect real data.

* The QKD layer is **emulated**. Key bits come from a seeded `numpy` random stream, which is neither secret nor
  cryptographically secure. The one-time pad and Wegman-Carter tag are real constructions, but their security
  statement holds only for uniformly random, secret, never-reused key bits, which a simulator does not supply.
* The leaf-to-EAN hop (a post-quantum KEM plus AEAD in the system model) is **not simulated**.
* Hash keys for the authentication tag come from a one-time trusted setup modeled by the same PRNG.

## Hardening already in the code

* Tag verification uses a constant-time comparison, and decryption happens only after a tag has verified.
* Decrypted payloads are length- and range-checked before use (`FormatError`), including the entry count and
  every index, so a malformed message cannot trigger an out-of-bounds write or a huge allocation.
* An authentication failure on either link drops that update (fail closed) and is counted in the run log.
* The CIFAR-10 batches are checked against pinned SHA-256 digests and parsed with an allow-listed unpickler, so a
  substituted cache file cannot run code when loaded.
* CI runs the tests, `bandit` and `pip-audit`; CodeQL and Dependabot are configured.

## Reporting a vulnerability

Please report privately through GitHub: **Security > Report a vulnerability** on this repository. If that is not
available, email sunil.gentyala@ieee.org with "QuantumShield-Edge security" in the subject. Do not open a public
issue for a vulnerability. Expect an acknowledgement within a few days.

In scope: flaws in the sealing, verification, parsing or data-loading code, and supply-chain problems in the
dependencies. Out of scope: the fact that emulated keys are not secure (stated above).
