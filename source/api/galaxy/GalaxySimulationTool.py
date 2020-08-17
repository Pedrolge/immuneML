import logging
import shutil
from glob import glob

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class GalaxySimulationTool:

    """
    GalaxySimulationTool is an alternative to running immuneML with the simulation instruction directly. It accepts a YAML specification file and a
    path to the output directory. It implants the signals in the dataset that was provided either as an existing dataset with a set of files or in
    the random dataset as described in the specification file.

    This tool is meant to be used as an endpoint for Galaxy tool that will create a Galaxy collection out of a dataset in immuneML format that can
    be readily used by other immuneML-based Galaxy tools.

    The specification supplied for this tool is identical to immuneML specification, except that it can include only one instruction which has to
    be of type 'Simulation':

    .. code-block: yaml

        definitions:
            datasets:
                my_synthetic_dataset:
                    format: RandomRepertoireDataset
                    params:
                        repertoire_count: 100
                        labels: {}
            motifs:
                my_simple_motif: # a simple motif without gaps or hamming distance
                  seed: AAA
                  instantiation: GappedKmer

                my_complex_motif: # complex motif containing a gap + hamming distance
                    seed: AA/A  # ‘/’ denotes gap position if present, if not, there’s no gap
                    instantiation:
                        GappedKmer:
                            min_gap: 1
                            max_gap: 2
                            hamming_distance_probabilities: # probabilities for each number of
                                0: 0.7                    # modification to the seed
                                1: 0.3
                            position_weights: # probabilities for modification per position
                                0: 1
                                1: 0 # note that index 2, the position of the gap,
                                3: 0 # is excluded from position_weights
                            alphabet_weights: # probabilities for using each amino acid in
                                A: 0.2      # a hamming distance modification
                                C: 0.2
                                D: 0.4
                                E: 0.2

            signals:
                my_signal:
                    motifs:
                    - my_simple_motif
                    - my_complex_motif
                    implanting: HealthySequence
                    sequence_position_weights:
                        109: 1
                        110: 2
                        111: 5
                        112: 1
            simulations:
                my_simulation:
                    my_implanting:
                        signals:
                        - my_signal
                        dataset_implanting_rate: 0.5
                        repertoire_implanting_rate: 0.25
        instructions:
            my_simulation_instruction: # user-defined name of the instruction
                type: Simulation # which instruction to execute
                dataset: my_dataset # which dataset to use for implanting the signals
                simulation: my_simulation # how to implanting the signals - definition of the simulation
                batch_size: 4 # how many parallel processes to use during execution
                export_formats: [AIRR] # in which formats to export the dataset, Pickle format will be added automatically
        output: # the output format
            format: HTML

    """

    def __init__(self, yaml_path, output_dir, **kwargs):
        Util.check_parameters(yaml_path, output_dir, kwargs, "Galaxy Simulation Tool")
        self.yaml_path = yaml_path
        self.result_path = output_dir if output_dir[-1] == '/' else f"{output_dir}/"

    def run(self):
        PathBuilder.build(self.result_path)
        self.prepare_specs()
        app = ImmuneMLApp(self.yaml_path, self.result_path)
        app.run()

        dataset_location = list(glob(self.result_path + "/*/exported_dataset/pickle/"))[0]

        shutil.copytree(dataset_location, self.result_path + 'result/')

        logging.info("GalaxySimulationTool: immuneML has finished and the signals were implanted in the dataset.")

    def prepare_specs(self):
        with open(self.yaml_path, "r") as file:
            specs = yaml.safe_load(file)

        assert "instructions" in specs, "GalaxySimulationTool: 'instructions' keyword missing from the specification."
        assert len(list(specs['instructions'].keys())) == 1, f"GalaxySimulationTool: multiple instructions were given " \
                                                             f"({str(list(specs['instructions'].keys()))[1:-1]}), but only one instruction of type " \
                                                             f"Simulation should be specified."
        instruction_name = list(specs['instructions'].keys())[0]
        instruction_type = specs['instructions'][instruction_name]['type']
        assert instruction_type == 'Simulation', f"GalaxySimulationTool: instruction type has to be 'Simulation', got {instruction_type} instead."

        if 'Pickle' not in specs['instructions'][instruction_name]['export_formats']:
            specs['instructions'][instruction_name]['export_formats'].append('Pickle')
            logging.info("GalaxySimulationTool: automatically adding 'Pickle' as export format...")

        Util.check_paths(specs, "GalaxySimulationTool")
        Util.update_result_paths(specs, self.result_path, self.yaml_path)