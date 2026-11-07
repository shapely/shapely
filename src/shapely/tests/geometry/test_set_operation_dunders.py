import operator

import numpy as np
import pytest

from shapely.geometry import box


class ReflectedSetOperations:
    def __ror__(self, other):
        return "reflected-or"

    def __rand__(self, other):
        return "reflected-and"

    def __rsub__(self, other):
        return "reflected-sub"

    def __rxor__(self, other):
        return "reflected-xor"


@pytest.mark.parametrize(
    "dunder",
    [
        "__or__",
        "__and__",
        "__sub__",
        "__xor__",
    ],
)
def test_set_operation_dunder_returns_notimplemented_for_unsupported_type(dunder):
    geometry = box(0, 0, 1, 1)

    result = getattr(geometry, dunder)(object())

    assert result is NotImplemented


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (operator.or_, "reflected-or"),
        (operator.and_, "reflected-and"),
        (operator.sub, "reflected-sub"),
        (operator.xor, "reflected-xor"),
    ],
)
def test_set_operation_falls_back_to_reflected_dunder(operation, expected):
    geometry = box(0, 0, 1, 1)
    other = ReflectedSetOperations()

    assert operation(geometry, other) == expected


@pytest.mark.parametrize(
    "operation",
    [
        operator.or_,
        operator.and_,
        operator.sub,
        operator.xor,
    ],
)
def test_set_operation_dunder_preserves_none_operand(operation):
    geometry = box(0, 0, 1, 1)

    assert operation(geometry, None) is None


@pytest.mark.parametrize(
    "operation",
    [
        operator.or_,
        operator.and_,
        operator.sub,
        operator.xor,
    ],
)
def test_set_operation_dunder_preserves_array_operand(operation):
    geometry = box(0, 0, 1, 1)
    others = np.array(
        [
            box(0.5, 0.5, 1.5, 1.5),
            box(0.25, 0.25, 0.75, 0.75),
        ],
        dtype=object,
    )

    result = operation(geometry, others)

    assert isinstance(result, np.ndarray)
    assert result.shape == others.shape