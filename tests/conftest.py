import os
import sys

import pytest

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)


def pytest_collection_modifyitems(config, items):
    from zuspec.be.xls.testing.xlstools import codegen_main, opt_main
    missing = {"xls": opt_main() is None,
               "xls_codegen": opt_main() is None or codegen_main() is None}
    for marker, tool in (("xls", "opt_main"), ("xls_codegen", "codegen_main")):
        if not missing[marker]:
            continue
        skip = pytest.mark.skip(reason=f"{tool} not found (set XLS_BIN or add it to PATH)")
        for item in items:
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)
