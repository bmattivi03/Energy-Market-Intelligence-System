"""Add the vendored GlocalIB repo to sys.path so its modules can be imported."""

import sys
from pathlib import Path

# Append (not insert-at-0): GlocalIB ships top-level ``utils`` and ``data``
# packages that collide with the project's ``src/utils`` / ``src/data``.
# Appending keeps the project packages ahead on sys.path while still exposing
# GlocalIB-only modules (``otherModel``, ``pypots``).
_GLOCAL_ROOT = Path(__file__).parent.parent.parent / "GlocalIB"
if str(_GLOCAL_ROOT) not in sys.path:
    sys.path.append(str(_GLOCAL_ROOT))
