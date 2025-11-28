from __future__ import annotations

import pandas as pd

from ..core.schema import ValidationSchema
from ..core.validator import ValidationBackend
from ..utils.reporting import ValidationReport


class PandasBackend(ValidationBackend[pd.DataFrame]):
    """Validates pandas DataFrames using the configured schema policies."""

    name = "pandas"

    def supports(self, data: object) -> bool:
        return isinstance(data, pd.DataFrame)

    def validate(self, data: pd.DataFrame, schema: ValidationSchema) -> ValidationReport:
        frame_report = schema.validate_dataframe(data)
        if schema.record_model is not None:
            record_report = schema.validate_records(data.to_dict(orient="records"))
            frame_report = frame_report.merge(record_report)
        return frame_report.with_metadata(backend=self.name, schema=schema.name)
