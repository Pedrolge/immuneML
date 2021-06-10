import datetime

import mlflow

from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_methods.SklearnMethod import SklearnMethod
from immuneML.output.Output import Output


class MLflow(Output):

    def __init__(self, url: str, experiment_name: str = "", log_params: bool = False, log_metrics: bool = False, log_model: bool = False):
        if experiment_name is None or not experiment_name:
            raise ValueError(
                f"Experiment name {experiment_name} must be defined for MLflow to be used. Please specify an experiment_name.")

        self.experiment_name = experiment_name
        self.mlflow_url = url
        self.should_log_params = log_params
        self.should_log_metrics = log_metrics
        self.should_log_model = log_model

        mlflow.set_tracking_uri(self.mlflow_url)
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)


    def set_context(self, context: dict):
        return self

    def publish_results(self, instruction_states):
        filtered_instruction_states = filter(lambda state: isinstance(state, TrainMLModelState), instruction_states)
        for state in filtered_instruction_states:
            with mlflow.start_run():
                self.publish_hp_optimization_results(state)

    def publish_hp_optimization_results(self, state: TrainMLModelState):
        for label in state.optimal_hp_items:
            method_key = state.optimal_hp_items[label].hp_setting.get_key()
            ml_method = state.optimal_hp_items[label].method

            if self.should_log_params:
                print(f"{datetime.datetime.now()}: Pusblishing parameters used to mlflow", flush=True)
                self.log_params(state.optimal_hp_items[label].hp_setting.ml_params)
            if self.should_log_metrics:
                print(f"{datetime.datetime.now()}: Pusblishing obtained metrics of model to mlflow", flush=True)
                self.log_metrics(state.optimal_hp_items[label].performance)
            if self.should_log_model and isinstance(ml_method, SklearnMethod):
                print(f"{datetime.datetime.now()}: Pusblishing obtained model to mlflow", flush=True)
                self.log_model(ml_method, method_key)

    def log_params(self, ml_params: dict):
        for param in ml_params:
            if isinstance(ml_params[param], dict):
                self.log_params(ml_params[param])
            else:
                mlflow.log_param(param, ml_params[param])

    def log_metrics(self, ml_performance: dict):
        for metric in ml_performance:
            if isinstance(ml_performance[metric], dict):
                self.log_metrics(ml_performance[metric])
            else:
                mlflow.log_metric(metric, ml_performance[metric])

    def log_model(self, ml_method: SklearnMethod, method_name: str):
            mlflow.sklearn.log_model(ml_method._get_ml_model(), method_name)
