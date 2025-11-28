"""Core validation primitives for data_guardian."""

from .schema import (
    ColumnSpec,
    SchemaBuilder,
    SchemaConverter,
    UnifiedSchema,
    ValidationSchema,
)
from .streaming import (
    StreamingResult,
    StreamingValidator,
    ValidationMetrics,
    validate_csv_streaming,
    validate_csv_streaming_sync,
)
from .validator import (
    AutoFixSuggestion,
    Backend,
    DataGuardianValidator,
    UnifiedValidator,
    ValidationBackend,
    ValidationErrorDetail,
    ValidationResult,
)

__all__ = [
    "AutoFixSuggestion",
    "Backend",
    "ColumnSpec",
    "DataGuardianValidator",
    "SchemaBuilder",
    "SchemaConverter",
    "StreamingResult",
    "StreamingValidator",
    "UnifiedSchema",
    "UnifiedValidator",
    "ValidationBackend",
    "ValidationMetrics",
    "ValidationSchema",
    "ValidationErrorDetail",
    "ValidationResult",
    "validate_csv_streaming",
    "validate_csv_streaming_sync",
]
