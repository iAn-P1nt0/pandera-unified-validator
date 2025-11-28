# data-guardian

Advanced data validation library that unifies Pydantic record validation with Pandera-powered
DataFrame checks. It ships with pluggable pandas and Polars backends so analytics teams can
reuse a single set of declarative rules across their favorite engines.

## Features

- 🔐 **Unified schemas** – describe record- and table-level constraints in one place.
- ⚙️ **Backend abstraction** – switch between pandas and Polars without rewriting rules.
- 🧪 **Modern tooling** – Hatch-based project with Ruff, Black, Mypy, Pytest, and Coverage.
- 📈 **Typed API** – Python 3.10+ with strict type hints and Protocol-driven backends.

## Installation

```bash
pip install data-guardian
```

For local development:

```bash
pip install hatch
hatch shell
```

## Quickstart

```python
from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Column, Check
from pydantic import BaseModel

from data_guardian import DataGuardianValidator, ValidationSchema, PandasBackend


class Order(BaseModel):
    order_id: int
    amount: float
    currency: str


dataframe_schema = pa.DataFrameSchema(
    {
        "order_id": Column(int, unique=True),
        "amount": Column(float, Check.greater_than_or_equal_to(0)),
        "currency": Column(str, Check.isin({"USD", "EUR"})),
    }
)

schema = ValidationSchema(
    name="orders",
    record_model=Order,
    dataframe_schema=dataframe_schema,
)

validator = DataGuardianValidator()
validator.register_backend(PandasBackend())

report = validator.validate(pd.DataFrame([{"order_id": 1, "amount": 9.99, "currency": "USD"}]), schema)
print(report.is_valid)  # True
print(report.metadata)
```

## Development

- Lint: `hatch run lint`
- Format: `hatch run fmt`
- Type-check: `hatch run typecheck`
- Tests: `hatch run test`

## Contributing

Issues and pull requests are welcome. Please run the full Hatch task suite before submitting
changes and include tests for new behavior.
