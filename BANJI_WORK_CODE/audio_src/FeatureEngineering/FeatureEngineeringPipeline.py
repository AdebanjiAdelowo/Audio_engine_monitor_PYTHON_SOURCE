from src.DataIO.Representation.Datum import Datum
from src.FeatureEngineering.FeatureExtraction.Interface import FeatureExtraction


class FeatureEngineeringPipeline:

    def __init__(self):
        """
        Initializes a new feature engineering pipeline.

        The feature engineering pipeline is a sequence of feature extraction blocks
        that are applied to a given datum to produce the final output.
        """
        # The feature engineering blocks are the main fea
        # ture extraction blocks
        # that are used to extract meaningful features from the given datum.
        self.__feature_engineering_blocks = []

        # The output blocks are the blocks that are used to post-process the
        # output of the feature engineering blocks and produce the final output.
        self.__output_blocks = []

    def add_block(self, block: FeatureExtraction) -> None:
        """
        Adds a feature engineering block to the pipeline.

        The block is added to the feature engineering blocks list
        and is executed in the order it was added.

        Args:
            block: The feature engineering block to add.
        """
        self.__feature_engineering_blocks.append(block)

    def add_output_block(self, block: FeatureExtraction) -> None:
        """
        Adds a output block to the pipeline.

        The output blocks are executed after the feature engineering blocks
        and are used to post-process the output of the feature engineering
        blocks and produce the final output.

        Args:
            block: The output block to add.
        """
        self.__output_blocks.append(block)

    def run(self, datum: Datum) -> Datum:
        """
        Runs the feature engineering pipeline.

        The pipeline is composed of a sequence of feature engineering blocks
        that are applied to the given datum to produce the final output.

        Args:
            datum: The datum to process.

        Returns:
            The processed datum.
        """
        # Run the feature engineering blocks
        for block in self.__feature_engineering_blocks:
            # Each block is executed in the order it was added
            datum = block.extract_features(datum)

        # Run the output blocks
        for block in self.__output_blocks:
            # The output blocks are executed after the feature engineering blocks
            # and are used to post-process the output of the feature engineering
            # blocks and produce the final output
            datum = block.extract_features(datum)

        return datum
