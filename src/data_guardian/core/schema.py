from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandera as pa
from pandera import DataFrameModel
from pydantic import BaseModel, ValidationError

from ..utils.reporting import ValidationReport

SchemaLike = pa.DataFrameSchema | type[DataFrameModel]


@dataclass(frozen=True)
class ValidationSchema:
    """Couples Pydantic models with Pandera dataframe schemas."""

    name: str
    record_model: type[BaseModel] | None = None
    dataframe_schema: SchemaLike | None = None

    def validate_records(self, rows: Iterable[Mapping[str, Any]]) -> ValidationReport:
        """Validate iterable of dictionary-like rows via the configured Pydantic model."""

        if self.record_model is None:
            return ValidationReport.ok(message="Record validation skipped", stage="records")

        errors: list[str] = []
        count = 0
        for count, row in enumerate(rows, start=1):
            try:
                self.record_model.model_validate(row)
            except ValidationError as exc:
                errors.append(str(exc))

        if errors:
            return ValidationReport(False, tuple(errors), (), {"stage": "records", "count": count})

        return ValidationReport.ok(message="Record validation passed", stage="records", count=count)

    def validate_dataframe(self, frame: Any) -> ValidationReport:
        """Validate a pandas-compatible dataframe via Pandera."""

        schema = self._materialize_schema()
        if schema is None:
            return ValidationReport.ok(message="Dataframe validation skipped", stage="dataframe")

        try:
            schema.validate(frame)
        except pa.errors.SchemaError as exc:
            return ValidationReport.from_exception(exc, stage="dataframe")

        size = getattr(frame, "shape", (None, None))[0]
        return ValidationReport.ok(message="Dataframe validation passed", stage="dataframe", rows=size)

    def validate_polars(self, frame: Any) -> ValidationReport:
        """Validate a Polars dataframe by converting it to pandas for Pandera inspection."""

        schema = self._materialize_schema()
        if schema is None:
            return ValidationReport.ok(message="Dataframe validation skipped", stage="dataframe")

        try:
            pandas_frame = frame.to_pandas()
        except AttributeError as exc:  # pragma: no cover - defensive
            return ValidationReport.from_exception(exc, stage="polars-conversion")

        return self.validate_dataframe(pandas_frame).with_metadata(source_backend="polars")

    def _materialize_schema(self) -> pa.DataFrameSchema | None:
        schema = self.dataframe_schema
        if schema is None:
            return None
        if isinstance(schema, pa.DataFrameSchema):
            return schema
        if isinstance(schema, type) and issubclass(schema, DataFrameModel):
            return schema.to_schema()
        raise TypeError("Unsupported schema type provided to ValidationSchema")
