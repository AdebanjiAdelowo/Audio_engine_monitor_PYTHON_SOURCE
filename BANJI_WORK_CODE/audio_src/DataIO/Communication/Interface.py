from abc import ABC, abstractmethod


class CommunicationInterface(ABC):

    @abstractmethod
    def read_sample(self) -> bytes:
        """Reads a sample from the communication interface.

        Returns:
            bytes: The sample read from the communication interface.
        """
        pass

    @abstractmethod
    def write_sample(self, sample: bytes) -> None:
        """Writes a sample to the communication interface.

        Args:
            sample (bytes): The sample to be written to the communication interface.
        """
        pass
