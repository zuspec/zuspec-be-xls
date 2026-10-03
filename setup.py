"""Supplies the one field pyproject.toml declares dynamic: the version, read
from src/zuspec/be/xls/__version__.py. Same mechanism as zuspec-ir-core's
setup.py, which explains why the shared zuspec-pybuild workflow needs it."""

import os

from setuptools import setup

proj_dir = os.path.dirname(os.path.abspath(__file__))


def _get_version():
    version_file = os.path.join(proj_dir, "src", "zuspec", "be", "xls", "__version__.py")
    glb = {}
    with open(version_file) as f:
        exec(f.read(), glb)
    return glb["VERSION"]


setup(version=_get_version())
