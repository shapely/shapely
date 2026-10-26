#!/usr/bin/env python3
"""Generate git-based version.

Note: This file is located next to meson.build or versioneer will not work.
"""

import os
import sys
from textwrap import dedent

import versioneer

sys.path.insert(0, "")


def write_version_info(path) -> None:
    """Write version Python file."""
    version = None
    git_version = None

    try:
        import _version

        version = _version.__version__
        git_version = _version.__git_version__
    except ImportError:
        version = versioneer.get_version()
        git_version = versioneer.get_versions()["full-revisionid"]
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
            version = versioneer.get_version()
        print(version)


if __name__ == "__main__":
    main()
