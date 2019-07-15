from . import ufuncs

__all__ = ["interpolate", "line_merge", "project", "shared_paths"]


def interpolate(line, normalize=False):
    if normalize:
        return ufuncs.interpolate_normalized(line)
    else:
        return ufuncs.interpolate(line)


def line_merge(lines):
    return ufuncs.line_merge(lines)


def project(line, other, normalize=False):
    if normalize:
        return ufuncs.project_normalized(line, other)
    else:
        return ufuncs.project(line, other)


def shared_paths(a, b=None, axis=0):
    if b is None:
        return ufuncs.shared_paths.reduce(a, axis=axis)
    else:
        return ufuncs.shared_paths(a, b)
