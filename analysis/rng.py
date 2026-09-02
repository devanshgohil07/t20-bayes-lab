"""
Reproducible seeding.

One master seed; every chain and every study derives its own stream from it, so a rerun
reproduces the numbers exactly.

Note on the key: Python's built-in hash() is salted per interpreter process, so using it
here would have made "seeded" runs differ from one run to the next. We use CRC-32 of the
UTF-8 bytes instead, which is stable across runs, machines and Python versions.
"""
import numpy as np
from zlib import crc32

MASTER_SEED = 20260901


def _key(*parts) -> int:
    return crc32("|".join(str(p) for p in parts).encode("utf-8"))


def chain_rng(model: str, chain: int, study: str = "main") -> np.random.Generator:
    """Independent generator for (study, model, chain)."""
    return np.random.default_rng([MASTER_SEED, _key(study, model, chain), chain])


def rng(tag: str) -> np.random.Generator:
    """Independent generator for a named one-off study."""
    return np.random.default_rng([MASTER_SEED, _key(tag)])
