from . import lib

__all__ = ["interpolate", "line_merge", "project", "shared_paths"]


def interpolate(line, normalize=False):
    if normalize:
        return lib.interpolate_normalized(line)
    else:
        return lib.interpolate(line)


def line_merge(lines):
    return lib.line_merge(lines)


def project(line, other, normalize=False):
    if normalize:
        return lib.project_normalized(line, other)
    else:
        return lib.project(line, other)


def shared_paths(a, b=None, axis=0):
    if b is None:
        return lib.shared_paths.reduce(a, axis=axis)
    else:
        return lib.shared_paths(a, b)
