"""Question-independent event caption allocation research pilot."""

from .config import Config
from .reproducibility import fix_seed

__all__ = ["Config", "fix_seed"]
