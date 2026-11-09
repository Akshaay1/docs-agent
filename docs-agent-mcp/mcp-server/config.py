"""Read numeric settings from the environment.

Every knob read here arrives as a string in the mcp-server-config ConfigMap
(docs-agent-mcp/charts/docs-agent/templates/configmap.yaml). A typo used to
crash the server at import time with a bare ValueError traceback that never
said which variable was wrong. These helpers fail fast and name the variable,
the expected type, the bad value, and the default that applies when the
variable is unset.
"""

from __future__ import annotations

import math
import os
from typing import TypeVar

T = TypeVar("T", int, float)


def env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Read an integer env var, or `default` when the variable is unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw, 10)
    except ValueError:
        raise _invalid(name, raw, "expected an integer", default) from None
    return _check_bounds(name, value, default, minimum, maximum)


def env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Read a float env var, or `default` when the variable is unset.

    float() accepts "nan" and "inf", which would flow into the reranker and
    silently poison every score, so non-finite values are rejected too.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise _invalid(name, raw, "expected a number", default) from None
    if not math.isfinite(value):
        raise _invalid(name, raw, "expected a finite number", default)
    return _check_bounds(name, value, default, minimum, maximum)


def _invalid(name: str, given: object, reason: str, default: object) -> ValueError:
    return ValueError(
        f"Invalid {name}={given!r}: {reason}. Fix the value or unset {name} to use the default {default}."
    )


def _check_bounds(name: str, value: T, default: T, minimum: T | None, maximum: T | None) -> T:
    if minimum is not None and value < minimum:
        raise _invalid(name, value, f"must be at least {minimum}", default)
    if maximum is not None and value > maximum:
        raise _invalid(name, value, f"must be at most {maximum}", default)
    return value
