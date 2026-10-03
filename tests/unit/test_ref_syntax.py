"""XT-1 / BX-1: our parser reads what XLS prints, and our printer prints it back
byte for byte."""
import os

import pytest

from zuspec.be.xls.printer import format_package
from zuspec.be.xls.testing.parser import parse_package
from zuspec.be.xls.testing.typecheck import check_package

REF = os.path.join(os.path.dirname(__file__), "data", "xls_ref")
REF_FILES = sorted(f for f in os.listdir(REF) if f.endswith(".ir"))


@pytest.mark.parametrize("name", REF_FILES)
def test_parse_check_print(name):
    with open(os.path.join(REF, name)) as f:
        text = f.read()
    pkg = parse_package(text)
    check_package(pkg)
    assert format_package(pkg) == text
