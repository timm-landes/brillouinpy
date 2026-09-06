import os
import sys

# The synthetic-data generators used by several tests live next to the
# runnable examples (examples/_synthetic_data.py), not in the package. Put that
# directory on sys.path so tests can `from _synthetic_data import ...`, the same
# way the scripts in examples/ and benchmarks/ do.
_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)
