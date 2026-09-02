"""Reproducible seeding. One master seed; every chain and every study derives
its own stream from it, so results are bit-for-bit repeatable."""
import numpy as np

MASTER_SEED = 20260901

def chain_rng(model: str, chain: int, study: str = "main") -> np.random.Generator:
    """Independent generator for (study, model, chain)."""
    key = abs(hash((study, model, chain))) % (2**31)
    return np.random.default_rng([MASTER_SEED, key, chain])

def rng(tag: str) -> np.random.Generator:
    return np.random.default_rng([MASTER_SEED, abs(hash(tag)) % (2**31)])
