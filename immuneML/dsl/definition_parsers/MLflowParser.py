from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.output.Output import Output
from immuneML.util.Logger import log
from immuneML.util.ReflectionHandler import ReflectionHandler


class MLflowParser:

    @staticmethod
    def parse(mlflow: dict, symbol_table: SymbolTable):
        if mlflow is None or len(mlflow) == 0:
            mlflow = {}

        for rep_id in mlflow.keys():
            symbol_table, mlflow[rep_id] = MLflowParser._parse_mlflow(rep_id, mlflow[rep_id], symbol_table)

        return symbol_table, mlflow

    @staticmethod
    @log
    def _parse_mlflow(key: str, params: dict, symbol_table: SymbolTable):
        valid_values = ReflectionHandler.all_nonabstract_subclass_basic_names(Output, "", "output/")
        mlflow_object, params = ObjectParser.parse_object(params, valid_values, "", "output/", "MLflowParser", key, builder=True,
                                                          return_params_dict=True)

        symbol_table.add(key, SymbolType.MLFLOW, mlflow_object)

        return symbol_table, params
