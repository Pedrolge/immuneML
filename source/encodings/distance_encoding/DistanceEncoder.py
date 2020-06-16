import pandas as pd

from scripts.specification_util import update_docs_per_mapping
from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from source.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison
from source.util import DistanceMetrics
from source.util.EncoderHelper import EncoderHelper
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class DistanceEncoder(DatasetEncoder):
    """
    Encodes a given RepertoireDataset as distance matrix, where the pairwise distance between each of the repertoires
    is calculated. The distance is calculated based on the presence/absence of elements defined under attributes_to_match.
    Thus, if attributes_to_match contains only 'sequence_aas', this means the distance between two repertoires is maximal
    if they contain the same set of sequence_aas, and the distance is minimal of none of the sequence_aas are shared between
    two repertoires.

    Attributes:

        distance_metric (:py:mod:`source.encodings.distance_encoding.DistanceMetricType`): The metric used to calculate the
            distance between two repertoires. Names of different distance metric types are allowed values in the specification.

        attributes_to_match: The attributes to consider when determining whether a sequence is present in both repertoires.
            Only the fields defined under attributes_to_match will be considered, all other fields are ignored.
            Valid values include any repertoire attribute (sequence, amino acid sequence, V gene etc).

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_distance_encoder:
            Distance:
                distance_metric: JACCARD
                sequence_batch_size: 1000
                attributes_to_match:
                    - sequence_aas
                    - v_genes
                    - j_genes
                    - chains
                    - region_types

    """

    def __init__(self, distance_metric: DistanceMetricType, attributes_to_match: list, sequence_batch_size: int, context: dict = None):
        self.distance_metric = distance_metric
        self.distance_fn = ReflectionHandler.import_function(self.distance_metric.value, DistanceMetrics)
        self.attributes_to_match = attributes_to_match
        self.sequence_batch_size = sequence_batch_size
        self.context = context

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def _prepare_parameters(distance_metric: str, attributes_to_match: list, sequence_batch_size: int, context: dict = None):
        valid_metrics = [metric.name for metric in DistanceMetricType]
        ParameterValidator.assert_in_valid_list(distance_metric, valid_metrics, "DistanceEncoder", "distance_metric")

        return {
            "distance_metric": DistanceMetricType[distance_metric.upper()],
            "attributes_to_match": attributes_to_match,
            "sequence_batch_size": sequence_batch_size,
            "context": context
        }

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = DistanceEncoder._prepare_parameters(**params)
            return DistanceEncoder(**prepared_params)
        else:
            raise ValueError("DistanceEncoder is not defined for dataset types which are not RepertoireDataset.")

    def build_distance_matrix(self, dataset: RepertoireDataset, params: EncoderParams, train_repertoire_ids: list):
        comparison = PairwiseRepertoireComparison(self.attributes_to_match, self.attributes_to_match, params["result_path"],
                                                  sequence_batch_size=self.sequence_batch_size)

        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        distance_matrix = comparison.compare(current_dataset, self.distance_fn, self.distance_metric.value)

        repertoire_ids = dataset.get_repertoire_ids()

        distance_matrix = distance_matrix.loc[repertoire_ids, train_repertoire_ids]

        return distance_matrix

    def build_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:

        lbl = ["repertoire_identifier"]
        lbl.extend(params["label_configuration"].get_labels_by_name())

        tmp_labels = dataset.get_metadata(lbl, return_df=True)
        tmp_labels = tmp_labels.iloc[pd.Index(tmp_labels['repertoire_identifier']).get_indexer(dataset.get_repertoire_ids())]
        tmp_labels = tmp_labels.to_dict("list")
        del tmp_labels["repertoire_identifier"]

        return tmp_labels

    def encode(self, dataset, params: EncoderParams) -> RepertoireDataset:

        train_repertoire_ids = EncoderHelper.prepare_training_ids(dataset, params)
        distance_matrix = self.build_distance_matrix(dataset, params, train_repertoire_ids)
        labels = self.build_labels(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix, labels=labels, example_ids=distance_matrix.index.values,
                                                   encoding=DistanceEncoder.__name__)

        self.store(encoded_dataset, params)

        return encoded_dataset

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"])

    @staticmethod
    def get_documentation():
        doc = str(DistanceEncoder.__doc__)

        valid_values = [metric.name for metric in DistanceMetricType]
        valid_values = str(valid_values)[1:-1].replace("'", "`")
        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Names of different distance metric types are allowed values in the specification.": f"Valid values are: {valid_values}.",
            "Valid values include any repertoire attribute (sequence, amino acid sequence, V gene etc).":
                f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
