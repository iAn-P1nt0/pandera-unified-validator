"""Public interface for the data_guardian package."""

from .core.schema import (
    ColumnSpec,
    SchemaBuilder,
    SchemaConverter,
    UnifiedSchema,
    ValidationSchema,
)
from .core.streaming import (
    StreamingResult,
    StreamingValidator,
    ValidationMetrics,
    validate_csv_streaming,
    validate_csv_streaming_sync,
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
    "StreamingResult",
    "StreamingValidator",
    "UnifiedSchema",
    "UnifiedValidator",
    "ValidationMetrics",
    "ValidationSchema",
    "ValidationErrorDetail",
    "ValidationResult",
    "validate_csv_streaming",
    "validate_csv_streaming_sync",
]

__version__ = "0.1.0"
