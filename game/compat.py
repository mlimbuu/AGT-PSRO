"""Cross-Python (3.9+) compatibility helpers, preserving newer behavior."""
import sys
from dataclasses import dataclass as _dataclass
from itertools import zip_longest


def dataclass(_cls=None, **kwargs):
    """Preserve stdlib dataclass slots on 3.10+; omit unsupported slots on 3.9."""
    if sys.version_info < (3, 10):
        kwargs.pop("slots", None)
    decorator = _dataclass(**kwargs)
    return decorator(_cls) if _cls is not None else decorator


def strict_zip(*iterables):
    """Python 3.9-compatible zip(..., strict=True): raise on unequal lengths."""
    marker = object()
    for row in zip_longest(*iterables, fillvalue=marker):
        if any(item is marker for item in row):
            raise ValueError("zip() arguments have unequal lengths")
        yield row
