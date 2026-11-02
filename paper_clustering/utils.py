"""Small helpers for retrying calls and consuming iterables in batches."""

from collections.abc import Callable, Iterable, Iterator
from functools import wraps
from itertools import islice
import logging
import math
import random
from time import sleep
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

logger = logging.getLogger(__name__)


def iter_batches(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield lists in input order, including a final partial batch.

    Consume at most one batch at a time. An empty input yields no batches.
    Validation occurs when iteration starts.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def retry(
    retries: int = 3,
    initial_backoff: float = 1,
    exponential_backoff: bool = True,
    jitter: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a synchronous call up to ``retries`` times after its first attempt.

    TypeError and ValueError propagate immediately. Each invocation starts with
    its own backoff; exponential backoff doubles the delay, and jitter adds a
    random value in [0, 1) when calculating the next delay after sleeping.
    """
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if not math.isfinite(initial_backoff) or initial_backoff < 0:
        raise ValueError("initial_backoff must be finite and non-negative")

    def retry_decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def retry_with_backoff_and_jitter(*args: P.args, **kwargs: P.kwargs) -> R:
            backoff = initial_backoff
            multiplier = 2 if exponential_backoff else 1
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except (TypeError, ValueError):
                    raise
                except Exception:
                    if attempt == retries:
                        raise
                    logger.exception(
                        "Failure in %s; %d retries remaining.",
                        func.__name__,
                        retries - attempt,
                    )
                    sleep(backoff)
                    backoff = backoff * multiplier
                    if jitter:
                        backoff += random.random()

        return retry_with_backoff_and_jitter

    return retry_decorator
