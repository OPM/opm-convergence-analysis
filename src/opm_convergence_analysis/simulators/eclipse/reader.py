from pathlib import Path
from typing import Union
from ...core.base_reader import BaseReader
from ...core.models import SimulationData


class EclipseReader(BaseReader):
    """
    Reader for Eclipse simulations.
    """

    def can_read(self, source: Union[str, Path]) -> bool:
        # Placeholder logic
        return False

    def read(self, source: Union[str, Path], **kwargs) -> SimulationData:
        raise NotImplementedError("Eclipse support is not yet implemented.")
