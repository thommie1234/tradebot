"""Config package — provides bootstrap() for consistent sys.path setup."""
import sys
from pathlib import Path


def bootstrap():
    """Find the project root by searching upward for config/ + engine/ markers."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'config').is_dir() and (parent / 'engine').is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return parent
    raise RuntimeError("Could not find project root")


REPO_ROOT = bootstrap()
