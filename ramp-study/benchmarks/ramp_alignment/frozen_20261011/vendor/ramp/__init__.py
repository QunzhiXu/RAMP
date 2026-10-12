"""Public API for the canonical two- and three-dataset RAMP implementation."""

from .ramp_ver2 import RampResult, SEED, __version__, fit_transform

__all__ = ["fit_transform", "RampResult", "SEED", "__version__"]
