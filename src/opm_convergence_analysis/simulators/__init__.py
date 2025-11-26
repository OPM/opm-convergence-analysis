from .opm import OPMReader
from .eclipse import EclipseReader
from .intersect import IntersectReader
from .jutuldarcy import JutulDarcyReader
from pathlib import Path
from typing import Union
from ..core.base_reader import BaseReader

__all__ = [
    "OPMReader",
    "EclipseReader",
    "IntersectReader",
    "JutulDarcyReader",
    "get_reader",
]


def get_reader(path: Union[str, Path]) -> BaseReader:
    """
    Detect and return the appropriate reader for the given path.

    Args:
        path: Path to simulation output file or directory.

    Returns:
        An instantiated reader capable of reading the path.

    Raises:
        ValueError: If no suitable reader is found.
    """
    readers = [OPMReader(), EclipseReader(), IntersectReader(), JutulDarcyReader()]

    for reader in readers:
        if reader.can_read(path):
            return reader

    raise ValueError(f"No suitable reader found for input: {path}")
