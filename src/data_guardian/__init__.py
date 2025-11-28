"""Public interface for the data_guardian package."""

from .core.schema import ValidationSchema
from .core.validator import DataGuardianValidator
from .backends.pandas_backend import PandasBackend
from .backends.polars_backend import PolarsBackend

__all__ = [
    "DataGuardianValidator",
    "PandasBackend",
    "PolarsBackend",
    "ValidationSchema",
]

__version__ = "0.1.0"
