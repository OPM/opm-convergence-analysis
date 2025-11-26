from pathlib import Path
from typing import Union
from ...core.base_reader import BaseReader
from ...core.models import SimulationData


class IntersectReader(BaseReader):
    """
    Reader for Intersect simulations.
    """

    def can_read(self, source: Union[str, Path]) -> bool:
        # Placeholder logic
        return False

    def read(self, source: Union[str, Path], **kwargs) -> SimulationData:
        raise NotImplementedError("Intersect support is not yet implemented.")
