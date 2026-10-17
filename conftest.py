"""
Pytest configuration for GestureTune test suite.
Adds project root to sys.path so 'src' and modules are discoverable.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

