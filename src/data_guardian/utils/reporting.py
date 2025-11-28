from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


@dataclass(frozen=True)
class ValidationReport:
    """Lightweight container describing the outcome of a validation run."""

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        *,
        message: str | None = None,
        stage: str | None = None,
        **metadata: Any,
    ) -> "ValidationReport":
        payload: MutableMapping[str, Any] = dict(metadata)
        if message:
            payload.setdefault("message", message)
        if stage:
            payload.setdefault("stage", stage)
        return cls(True, (), (), payload)

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        stage: str | None = None,
    ) -> "ValidationReport":
        message = str(exc).strip() or exc.__class__.__name__
        payload: MutableMapping[str, Any] = {}
        if stage:
            payload["stage"] = stage
        return cls(False, (message,), (), payload)

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        metadata = {**self.metadata, **other.metadata}
        return ValidationReport(
            self.is_valid and other.is_valid,
            self.errors + other.errors,
            self.warnings + other.warnings,
            metadata,
        )

    def with_metadata(self, **metadata: Any) -> "ValidationReport":
        return ValidationReport(self.is_valid, self.errors, self.warnings, {**self.metadata, **metadata})
