#!/usr/bin/env python3
"""Generate git-based version."""

import os
import sys
from textwrap import dedent

sys.path.insert(0, "")
here = os.path.abspath(os.path.dirname(__file__))


def _get_version_info():
    """Get the source code management version info using vcs-versioning."""
    from vcs_versioning import Configuration

    config = Configuration.from_file(os.path.join(here, "pyproject.toml"))
    workdir = config.discover_workdir()
    version_info = workdir.get_scm_version()
    return version_info


def get_version():
    """Get the version string."""
    version_info = _get_version_info()
    return version_info.format()


def get_git_revision():
    """Get the git revision (or node)."""
    version_info = _get_version_info()
    return version_info.node


def write_version_info(path) -> None:
    """Write version Python file."""
    version = None
    git_version = None

    try:
        import _version

        version = _version.__version__
        git_version = _version.__git_version__
    except ImportError:
        version = get_version()
        git_version = get_git_revision()
    content = dedent(
        f'''\
        """Module to show version information for the installed package."""

        __version__ = "{version}"
        __git_version__ = "{git_version}"
        '''
    )
    if os.environ.get("MESON_DIST_ROOT"):
        path = os.path.join(os.environ.get("MESON_DIST_ROOT"), path)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def main() -> None:
    """CLI entry point."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--outfile",
        help="Path to write version info to",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Whether to print out the version",
    )
    args = parser.parse_args()

    if args.outfile:
        if not args.outfile.endswith(".py"):
            raise ValueError(
                f"Output file must be a Python file. "
                f"Got: {args.outfile} as filename instead"
            )

        write_version_info(args.outfile)

    if args.print:
        try:
            import _version

            version = _version.__version__
        except ImportError:
            version = get_version()
        print(version)


if __name__ == "__main__":
    main()
