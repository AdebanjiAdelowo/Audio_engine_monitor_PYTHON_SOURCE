from abc import ABC, abstractmethod
import numpy as np


class Filter(ABC):

    @abstractmethod
    def apply_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the filter to the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The signal to filter.
        sample_rate : int
            The sample rate of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.
        """
        pass
