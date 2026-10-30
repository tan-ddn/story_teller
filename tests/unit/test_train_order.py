import random
from collections.abc import Iterator
from typing import Any

from training.censor.train import _reproducible_order

N = 10


def shuffled(*_: Any, **__: Any) -> Iterator[int]:
    """Stands in for mlx-lm's iterate_batches(loop=False): one epoch from the global generator."""
    yield from random.sample(range(N), N)


def take(it: Iterator[int], n: int) -> list[int]:
    return [next(it) for _ in range(n)]


def test_same_seed_same_order_across_epochs() -> None:
    a = _reproducible_order(shuffled, random.seed, 0, 0)(loop=True)
    random.seed(123)  # whatever else touched the global generator does not matter
    b = _reproducible_order(shuffled, random.seed, 0, 0)(loop=True)
    first = take(a, 2 * N)
    assert first == take(b, 2 * N)
    assert sorted(first[:N]) == list(range(N))  # one full epoch, then the next
    assert first[:N] != first[N:]  # each epoch has its own shuffle


def test_resume_continues_the_same_order() -> None:
    full = take(_reproducible_order(shuffled, random.seed, 1, 0)(loop=True), 2 * N)
    resumed = take(_reproducible_order(shuffled, random.seed, 1, 7)(loop=True), 2 * N - 7)
    assert resumed == full[7:]


def test_other_seed_other_order() -> None:
    a = take(_reproducible_order(shuffled, random.seed, 0, 0)(loop=True), N)
    b = take(_reproducible_order(shuffled, random.seed, 1, 0)(loop=True), N)
    assert a != b


def test_validation_batches_untouched() -> None:
    calls: list[int] = []
    wrapped = _reproducible_order(shuffled, calls.append, 0, 3)
    assert len(list(wrapped(loop=False))) == N
    assert calls == []
