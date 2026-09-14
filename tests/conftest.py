import importlib
import os
import sys

import pytest

# The synthetic-data generators used by several tests live next to the
# runnable examples (examples/_synthetic_data.py), not in the package. Put that
# directory on sys.path so tests can `from _synthetic_data import ...`, the same
# way the scripts in examples/ and benchmarks/ do.
_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)


def require_optional(module_name):
    """Import an optional test dependency, or skip/fail depending on the environment.

    Behaves like ``pytest.importorskip`` by default: a missing (or otherwise
    unimportable, e.g. a ``SyntaxError`` from a backend that doesn't support this
    Python version) module skips the test. When the ``BRILLOUINPY_REQUIRE_OPTIONAL``
    environment variable is set to ``"1"``, a missing module is a hard failure
    instead, so a CI job that means to exercise every optional backend can't pass
    with them silently skipped.

    Parameters
    ----------
    module_name : str
        The importable name of the optional module (e.g. ``"brimfile"``).

    Returns
    -------
    module
        The imported module.
    """
    try:
        return importlib.import_module(module_name)
    except (ImportError, SyntaxError) as exc:
        message = f"{module_name} not importable: {exc}"
        if os.environ.get("BRILLOUINPY_REQUIRE_OPTIONAL") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=True)
