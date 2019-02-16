import copy
import random

from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.simulation.implants.ImplantAnnotation import ImplantAnnotation
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from source.simulation.signal_implanting_strategy.sequence_implanting.SequenceImplantingStrategy import \
    SequenceImplantingStrategy


class HealthySequenceImplanting(SignalImplantingStrategy):
    """
    Class for implanting a signal into a repertoire:
        - always chooses only sequences in which no signal has been implanted to implant the new signal
        - sequence_position_weights define the probability that the signal will be implanted at the
            certain position in the receptor_sequence
        - if sequence_position_weights are not set, then SequenceImplantingStrategy will make all of the positions
            equally likely for each receptor_sequence
    """

    def __init__(self, sequence_implanting_strategy: SequenceImplantingStrategy, sequence_position_weights: dict = None):
        self.sequence_implanting_strategy = sequence_implanting_strategy
        self.sequence_position_weights = sequence_position_weights

    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal) -> Repertoire:
        max_motif_length = self._calculate_max_motif_length(signal)
        sequences_to_be_processed, other_sequences = self._choose_sequences_for_implanting(repertoire,
                                                                                            repertoire_implanting_rate,
                                                                                            max_motif_length)
        processed_sequences = self._implant_in_sequences(sequences_to_be_processed, signal)
        sequences = other_sequences + processed_sequences
        metadata = self._build_new_metadata(repertoire.metadata)
        new_repertoire = self._build_new_repertoire(sequences, metadata, signal)

        return new_repertoire

    def _build_new_metadata(self, metadata: RepertoireMetadata) -> RepertoireMetadata:
        new_metadata = copy.deepcopy(metadata) if metadata is not None else RepertoireMetadata()

        if new_metadata.sample is None:
            new_metadata.sample = Sample("", custom_params={})

        return new_metadata

    def _calculate_max_motif_length(self, signal):
        max_motif_length = max([motif.get_max_length() for motif in signal.motifs])
        return max_motif_length

    def _build_new_repertoire(self, sequences, repertoire_metadata, signal) -> Repertoire:
        if repertoire_metadata is not None:
            metadata = copy.deepcopy(repertoire_metadata)
        else:
            metadata = RepertoireMetadata()

        # when adding implant to a repertoire, only signal id is stored:
        # more detailed information is available in each receptor_sequence
        # (specific motif and motif instance)
        implant = ImplantAnnotation(signal_id=signal.id)
        metadata.add_implant(implant)
        repertoire = Repertoire(sequences=sequences, metadata=metadata)

        return repertoire

    def _implant_in_sequences(self, sequences_to_be_processed: list, signal):
        assert self.sequence_implanting_strategy is not None, "HealthySequenceImplanting: add receptor_sequence implanting strategy when creating a HealthySequenceImplanting object."

        sequences = []
        for sequence in sequences_to_be_processed:
            processed_sequence = self.implant_in_sequence(sequence, signal)
            sequences.append(processed_sequence)

        return sequences

    def _choose_sequences_for_implanting(self, repertoire: Repertoire, repertoire_implanting_rate: float, max_motif_length: int):
        number_of_sequences_to_implant = int(repertoire_implanting_rate * len(repertoire.sequences))
        unusable_sequences = []
        unprocessed_sequences = []

        for sequence in repertoire.sequences:
            if sequence.annotation is not None and sequence.annotation.implants is not None and len(sequence.annotation.implants) > 0:
                unusable_sequences.append(sequence)
            elif len(sequence.get_sequence()) <= max_motif_length:
                unusable_sequences.append(sequence)
            else:
                unprocessed_sequences.append(sequence)

        assert number_of_sequences_to_implant <= len(unprocessed_sequences), "HealthySequenceImplanting: there are not enough sequences in the repertoire to provide given repertoire infection rate. Reduce repertoire infection rate to proceed."

        random.shuffle(unprocessed_sequences)
        sequences_to_be_infected = unprocessed_sequences[:number_of_sequences_to_implant]
        other_sequences = unusable_sequences + unprocessed_sequences[number_of_sequences_to_implant:]

        return sequences_to_be_infected, other_sequences

    def implant_in_sequence(self, sequence: ReceptorSequence, signal) -> ReceptorSequence:
        assert self.sequence_implanting_strategy is not None, "HealthySequenceImplanting: set SequenceImplantingStrategy in HealtySequenceImplanting object before calling implant_in_sequence method."

        motif = random.choice(signal.motifs)
        motif_instance = motif.instantiate_motif()
        new_sequence = self.sequence_implanting_strategy.implant(sequence=sequence,
                                                                 signal={"signal_id": signal.id,
                                                                         "motif_id": motif.id,
                                                                         "motif_instance": motif_instance},
                                                                 sequence_position_weights=self.sequence_position_weights)
        return new_sequence
