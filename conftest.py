import os
import sys

# Make `submission` and `harness` importable as top-level packages when
# pytest is invoked from anywhere inside this repository.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
