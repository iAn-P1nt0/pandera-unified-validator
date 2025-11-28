from __future__ import annotations

import pytest

from data_guardian import DataGuardianValidator
from data_guardian.backends import PandasBackend


@pytest.fixture()
def pandas_validator() -> DataGuardianValidator:
    validator = DataGuardianValidator()
    validator.register_backend(PandasBackend())
    return validator
