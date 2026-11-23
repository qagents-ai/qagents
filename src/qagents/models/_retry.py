"""Parse the provider's earliest allowed retry, in seconds."""

import math
import time
from collections.abc import Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    for key, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(key)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
        except ValueError:
            if key == "retry-after":
                try:
                    when = parsedate_to_datetime(value)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=UTC)
                    return max(0.0, when.timestamp() - time.time())
                except (ValueError, TypeError, OverflowError, OSError):
                    pass
            continue
        if math.isfinite(delay) and delay >= 0:
            return delay
    return None
