from __future__ import annotations

import polars as pl

from ..core.schema import ValidationSchema
from ..core.validator import ValidationBackend
from ..utils.reporting import ValidationReport


class PolarsBackend(ValidationBackend[pl.DataFrame]):
    """Validates Polars DataFrames by delegating checks through Pandera."""

    name = "polars"

    def supports(self, data: object) -> bool:
        return isinstance(data, pl.DataFrame)

    def validate(self, data: pl.DataFrame, schema: ValidationSchema) -> ValidationReport:
        frame_report = schema.validate_polars(data)
        if schema.record_model is not None:
            record_report = schema.validate_records(data.to_dicts())
            frame_report = frame_report.merge(record_report)
        return frame_report.with_metadata(backend=self.name, schema=schema.name)
