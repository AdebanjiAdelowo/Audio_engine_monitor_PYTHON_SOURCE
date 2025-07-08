from enum import Enum
from typing import List, Optional, Dict, Any


class DatumType(Enum):
    NUMERIC = 1
    AUDIO = 2
    IMAGE = 3
    VIDEO = 4
    TEXT = 5


class RawDatumKey(Enum):
    AUDIO_ARRAY = 'audio_array'
    SAMPLE_RATE = 'sample_rate'


class DerivedDataKey(Enum):
    ENGINE_STATE = 'engine_state'
    DECIMATED_AUDIO_RATE = 'decimated_audio_rate'
    RPM = 'RPM'
    DECIMATED_AUDIO = 'decimated_audio'
    FREQUENCY_PEAK = 'frequency_peak'
    FREQUENCY_PEAK_FILTERED_AUDIO = 'frequency_peak_filtered_audio'


class Datum:
    def __init__(self, datum_type: DatumType) -> None:
        """
        :param datum_type: The type of the datum
        """
        self.__raw_datum: Dict[RawDatumKey, Any] = {}
        """
        A dictionary containing the raw data of the Datum.
        The keys are the RawDatumKey and the values are the raw data.
        """
        self.__datum_type: DatumType = datum_type
        """
        The type of the Datum.
        """
        self.__derived_data: Optional[Dict[DerivedDataKey, Any]] = {}
        """
        A dictionary containing the derived data of the Datum.
        The keys are the DerivedDataKey and the values are the derived data.
        """
        self.__labels: Optional[List] = []
        """
        A list of the labels of the Datum.
        """
        self.__labels_names: Optional[List[str]] = []
        """
        A list of the names of the labels of the Datum.
        """

    def add_raw_datum(self, key: RawDatumKey, value: Any) -> None:
        """
        Adds a raw datum to the Datum.

        :param key: The key of the raw datum.
        :param value: The value of the raw datum.
        """
        self.__raw_datum[key] = value

    def remove_raw_datum(self, key: RawDatumKey) -> None:
        """
        Removes a raw datum from the Datum.

        :param key: The key of the raw datum to be removed.
        """
        del self.__raw_datum[key]

    def get_raw_datum(self, key: RawDatumKey) -> Any:
        """
        Gets a raw datum from the Datum.

        :param key: The key of the raw datum.
        :return: The value of the raw datum.
        """
        return self.__raw_datum[key]

    def get_raw_datum_keys(self) -> List[RawDatumKey]:
        """
        Gets a list of all the raw data keys of the Datum.

        :return: A list of RawDatumKey.
        """
        return list(self.__raw_datum.keys())

    def add_derived_data(self, key: DerivedDataKey, value: Any) -> None:
        """
        Adds a derived datum to the Datum.

        :param key: The key of the derived datum.
        :param value: The value of the derived datum.
        """
        self.__derived_data[key] = value

    def remove_derived_data(self, key: DerivedDataKey) -> None:
        """
        Removes a derived datum from the Datum.

        :param key: The key of the derived datum to be removed.
        """
        del self.__derived_data[key]

    def get_derived_data(self, key: DerivedDataKey) -> Any:
        """
        Gets a derived datum from the Datum.

        :param key: The key of the derived datum.
        :return: The value of the derived datum.
        """
        return self.__derived_data[key]

    def get_derived_data_keys(self) -> List[DerivedDataKey]:
        """
        Gets a list of all the derived data keys of the Datum.

        :return: A list of DerivedDataKey.
        """
        return list(self.__derived_data.keys())

    def set_labels(self, labels: List) -> None:
        """
        Sets the labels of the Datum.

        :param labels: The labels to be set.
        """
        self.__labels = labels

    def set_labels_names(self, labels_names: List[str]) -> None:
        """
        Sets the names of the labels of the Datum.

        :param labels_names: The names of the labels to be set.
        """
        self.__labels_names = labels_names

    def get_labels(self) -> List:
        """
        Gets the labels of the Datum.

        :return: A list of the labels of the Datum.
        """
        return self.__labels

    def get_labels_names(self) -> List[str]:
        """
        Gets the names of the labels of the Datum.intuos04
        

        :return: A list of strings representing the names of the labels.
        """
        return self.__labels_names
