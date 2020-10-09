from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.util.ImportHelper import ImportHelper


class IGoRImport(DataImport):
    """
    Imports data generated by IGoR simulations into a RepertoireDataset. Assumes one file per repertoire.
    Note that you should run IGoR with the --CDR3 option specified, this tool imports the generated CDR3 files.
    Sequences with missing anchors are not imported.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_igor_dataset:
            format: IGoR
            params:
                # required parameters:
                metadata_file: path/to/metadata.csv
                path: path/to/directory/with/repertoire/files/
                result_path: path/where/to/store/imported/repertoires/
                # optional parameters (if not specified the values bellow will be used):
                import_missing_anchors: False # if set to False, erroneous sequences, characterized by 0 in the anchors_found columns, are removed
                import_with_stop_codon: False # whether to import sequences with stop codon
                import_out_of_frame: False # whether out of frame sequences should be imported
                separator: "," # column separator of the input file
                columns_to_load: ["seq_index", "nt_CDR3", "anchors_found", "is_inframe"]
                region_definition: "IMGT" # which CDR3 definition to use - IMGT option means removing first and last amino acid as IGoR uses IMGT junction
                region_type: "CDR3"
                column_mapping: # IGoR column names -> immuneML repertoire fields
                  nt_CDR3: sequences
                  seq_index: sequence_identifiers

    """
    CODON_TABLE = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
        }

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> RepertoireDataset:
        igor_params = DatasetImportParams.build_object(**params)
        return ImportHelper.import_repertoire_dataset(IGoRImport.preprocess_repertoire, igor_params, dataset_name)

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams):
        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)

        if "counts" not in df.columns:
            df["counts"] = 1

        df = df[df.anchors_found == "1"]

        if not params.import_out_of_frame:
            df = df[df.is_inframe == "1"]

        df["sequence_aas"] = df["sequences"].apply(IGoRImport.translate_sequence)

        if not params.import_with_stop_codon:
            no_stop_codon = ["*" not in seq for seq in df.sequence_aas]
            df = df[no_stop_codon]

        ImportHelper.junction_to_cdr3(df, params.region_definition, params.region_type)

        # chain or at least receptorsequence?

        return df

    @staticmethod
    def translate_sequence(nt_seq):
        '''
        Code inspired by: https://github.com/prestevez/dna2proteins/blob/master/dna2proteins.py
        '''
        aa_seq = []
        end = len(nt_seq) - (len(nt_seq) % 3) - 1
        for i in range(0, end, 3):
            codon = nt_seq[i:i + 3]
            if codon in IGoRImport.CODON_TABLE:
                aminoacid = IGoRImport.CODON_TABLE[codon]
                aa_seq.append(aminoacid)
            else:
                aa_seq.append("_")
        return "".join(aa_seq)

