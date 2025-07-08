from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class DataPoint(ABC):

    @abstractmethod
    def get_unix_timestamp(self) -> float:
        """
        Get the time at which the data point was recorded, in seconds since the Unix epoch.

        Returns:
            float: The time at which the data point was recorded, in seconds since the Unix epoch.
        """
        pass

    @abstractmethod
    def get_value(self) -> Union[np.ndarray, float, int]:
        """
        Get the value of the data point.

        This can be any type that is appropriate for the data point. It could be a single number, a numpy
        array of numbers, or something else.

        Returns:
            Union[np.ndarray, float, int]: The value of the data point.
        """
        pass

    @abstractmethod
    def set_unix_timestamp(self, timestamp: float):
        """
        Set the time at which the data point was recorded, in seconds since the Unix epoch.

        Parameters:
            timestamp (float): The time at which the data point was recorded, in seconds since the Unix epoch.
        """
        pass

    @abstractmethod
    def set_value(self, value: Union[np.ndarray, float, int]):
        """
        Set the value of the data point.

        This can be any type that is appropriate for the data point. It could be a single number, a numpy
        array of numbers, or something else.

        Parameters:
            value (Union[np.ndarray, float, int]): The value of the data point.
        """
        pass
