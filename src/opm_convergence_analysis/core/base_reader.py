from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
from .models import SimulationData


class BaseReader(ABC):
    """
    Abstract base class for simulator convergence data readers.
    """

    @abstractmethod
    def read(self, source: Union[str, Path], **kwargs) -> SimulationData:
        """
        Read data from the source and return a SimulationData object.

        Args:
            source: Path to the main output file or directory.
            **kwargs: Additional simulator-specific arguments.

        Returns:
            SimulationData: The parsed data in generic format.
        """
        pass

    @abstractmethod
    def can_read(self, source: Union[str, Path]) -> bool:
        """
        Check if this reader can handle the given source.
        """
        pass
