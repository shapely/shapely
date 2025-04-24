import warnings

import pytest

from shapely.decorators import deprecate_positional


@deprecate_positional(["b", "c"])
def func_two(a, b=2, c=3):
    return a, b, c


@deprecate_positional(["b", "c", "d"])
def func_three(a, b=1, c=2, d=3):
    return a, b, c, d


@deprecate_positional(["b", "d"])
def func_noncontig(a, b=1, c=2, d=3):
    return a, b, c, d


@deprecate_positional(["b"], category=UserWarning)
def func_custom_category(a, b=1):
    return a, b


@deprecate_positional(["b"])
def func_varargs(a, b=1, *args):
    return a, b, args


@deprecate_positional([])
def func_no_deprecations(a, b=1):
    return a, b


def test_all_kwargs_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        assert func_two(a=10, b=20, c=30) == (10, 20, 30)
        assert not caught


def test_only_required_arg_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        assert func_two(1) == (1, 2, 3)
        assert not caught


def test_single_positional_warning():
    with warnings.catch_warnings(record=True) as caught:
        out = func_two(1, 4)
        assert out == (1, 4, 3)
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "positional argument `b` for `func_two` is deprecated." in msg


def test_multiple_positional_warning():
    with warnings.catch_warnings(record=True) as caught:
        out = func_two(1, 4, 5)
        assert out == (1, 4, 5)
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "positional arguments `b` and `c` for `func_two` are deprecated." in msg


def test_three_positional_warning_oxford_comma():
    with warnings.catch_warnings(record=True) as caught:
        out = func_three(1, 2, 3, 4)
        assert out == (1, 2, 3, 4)
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert (
            "positional arguments `b`, `c`, and `d` for `func_three` are deprecated."
            in msg
        )


def test_noncontiguous_partial_warning():
    with warnings.catch_warnings(record=True) as caught:
        out = func_noncontig(1, 2, 3)
        assert out == (1, 2, 3, 3)
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "positional argument `b` for `func_noncontig` is deprecated." in msg


def test_noncontiguous_full_warning():
    with warnings.catch_warnings(record=True) as caught:
        out = func_noncontig(1, 2, 3, 4)
        assert out == (1, 2, 3, 4)
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert (
            "positional arguments `b` and `d` for `func_noncontig` are deprecated."
            in msg
        )


def test_custom_warning_category():
    with warnings.catch_warnings(record=True) as caught:
        out = func_custom_category(1, 2)
        assert out == (1, 2)
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)


def test_func_no_deprecations_never_warns():
    with warnings.catch_warnings(record=True) as caught:
        out = func_no_deprecations(7, 8)
        assert out == (7, 8)
        assert not caught


def test_missing_required_arg_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(TypeError):
            func_two()  # missing required 'a'
        assert not caught


def test_unknown_keyword_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(TypeError):
            func_two(1, 4, d=5)  # unknown keyword 'd'
        assert not caught


def test_varargs_behavior_and_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        out = func_varargs(1, 2, 3, 4)
        assert out == (1, 2, (3, 4))
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "positional argument `b` for `func_varargs` is deprecated." in msg

    with warnings.catch_warnings(record=True) as caught:
        out = func_varargs(1)
        assert out == (1, 1, ())
        assert not caught


def test_repeated_warnings():
    with warnings.catch_warnings(record=True) as caught:
        func_two(1, 4, 5)
        func_two(1, 4, 5)
        assert len(caught) == 2
        assert str(caught[0].message) == str(caught[1].message)
