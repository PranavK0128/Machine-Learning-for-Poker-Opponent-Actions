"""Shared repaired Stage 21 heads-up no-limit Hold'em baseline core.

The package is intentionally dependency-light so the exact same source bytes can
be bundled into River, Turn, Flop, and Preflop Colab input packages.
"""

from .common import ALGORITHMS, SCIENCE_SEEDS, WORK_LANES_PER_PRODUCER

__all__ = ["ALGORITHMS", "SCIENCE_SEEDS", "WORK_LANES_PER_PRODUCER"]
