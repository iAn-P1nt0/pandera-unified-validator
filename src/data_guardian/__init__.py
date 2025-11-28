"""Public interface for the data_guardian package."""

from .core.schema import (
    ColumnSpec,
    SchemaBuilder,
    SchemaConverter,
    UnifiedSchema,
    ValidationSchema,
)
from .core.validator import (
    AutoFixSuggestion,
    DataGuardianValidator,
    UnifiedValidator,
    ValidationErrorDetail,
    ValidationResult,
)
from .backends.pandas_backend import PandasBackend
from .backends.polars_backend import PolarsBackend

__all__ = [
    "AutoFixSuggestion",
    "ColumnSpec",
    "DataGuardianValidator",
    "PandasBackend",
    "PolarsBackend",
    "SchemaBuilder",
    "SchemaConverter",
    "UnifiedSchema",
    "UnifiedValidator",
    "ValidationSchema",
    "ValidationErrorDetail",
    "ValidationResult",
]

__version__ = "0.1.0"
