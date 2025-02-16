import mlflow

# Expected logging parameters for MLflow
expected_mlflow_logging_params = {
    "tracking_uri": None,           # The URI of the MLflow tracking server
    "experiment_name": None,        # Name of the MLflow experiment
    "run_name": None,               # Name of the MLflow run
    "parameters": {},               # Dictionary of parameters to log, such as hyperparameters of the model (not metrics)
}


class MLFlow_LoggingManager:
    """
    A class for managing MLflow logging, including validation of logging parameters,
    setup of the tracking environment, and logging of parameters, models, and run information.
    """

    def __init__(self, mlflow_logging_parameters: dict):
        """
        Initialize the LoggingManager with logging parameters.

        Args:
            mlflow_logging_parameters (dict): The dictionary containing MLflow logging parameters.
        """
        self._mlflow_logging_parameters = mlflow_logging_parameters
        self._validate_logging_parameters()

    def _validate_logging_parameters(self):
        """
        Validate the logging parameters to ensure all expected keys are present and correctly formatted.

        Args:
            mlflow_logging_parameters (dict): The dictionary of logging parameters to validate.

        Returns:
            bool: True if the logging parameters are valid.

        Raises:
            ValueError: If a required key is missing or the parameters are invalid.
        """
        for key in expected_mlflow_logging_params.keys():
            if key not in self._mlflow_logging_parameters.keys():
                raise ValueError(f"MLflow logging parameter '{key}' is missing.")
        
        # Ensure parameters and metrics is a dictionary
        if not isinstance(self._mlflow_logging_parameters["parameters"], dict):
            raise ValueError("'parameters' must be a dictionary.")
        
    def set_up_MLflow_tracking(self):
        """
        Set up the MLflow tracking environment using the stored logging parameters.
        """
        try:
            mlflow.set_tracking_uri(self._mlflow_logging_parameters["tracking_uri"])
            mlflow.set_experiment(self._mlflow_logging_parameters["experiment_name"])
            
            # Enable autologging: MLflow will log the metrics at the end of each epoch. 
            # The matrics that will be logged are the ones used to compile the model
            mlflow.autolog(every_n_iter=1)  
        
        except Exception as e:
            raise RuntimeError(f"Error setting up MLflow tracking: {e}")

    def log_parameters(self):
        """
        Log parameters to MLflow from the stored logging parameters.
        """
        for parameter, value in self._mlflow_logging_parameters["parameters"].items():
            mlflow.log_param(parameter, value)
            
    def start_logging(self, run_id=None):
        """
        Start an MLflow run, set up tracking, and log parameters.

        Args:
            run_id (str, optional): The ID of an existing MLflow run to resume. Defaults to None.
        """
        self.set_up_MLflow_tracking()
        mlflow.start_run(run_name=self._mlflow_logging_parameters["run_name"], run_id=run_id)
                
        self._mlflow_logging_parameters["run_id"] = mlflow.active_run().info.run_id
        
    def log_evaluation_metrics(self, metrics: dict):
        """
        Log evaluation metrics to MLflow.

        Args:
            metrics (dict): A dictionary of metrics to log.
        """
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
    def end_logging(self):
        """
        End the current MLflow run.
        """
        mlflow.end_run()

    def log_model(self, model, artifact_path: str = 'models'):
        """
        Log a trained model to MLflow.

        Args:
            model (any): The trained model object to log. This can be a TensorFlow, PyTorch, or any compatible MLflow model.
            artifact_path (str): The path in the MLflow artifact store to save the model.
        """
        try:
            mlflow.tensorflow.log_model(model, artifact_path)
        except Exception as e:
            raise RuntimeError(f"Error logging the model to MLflow: {e}")

    # Getters
    def get_logging_parameters(self) -> dict:
        """
        Get the stored logging parameters.

        Returns:
            dict: The MLflow logging parameters.
        """
        return self._mlflow_logging_parameters

    def get_expected_logging_params(self) -> dict:
        """
        Get the expected logging parameters format.

        Returns:
            dict: The expected MLflow logging parameters.
        """
        return expected_mlflow_logging_params

    def get_active_run_id(self) -> str:
        """
        Get the ID of the active MLflow run.

        Returns:
            str: The ID of the active MLflow run.
        """
        run_id = self._mlflow_logging_parameters["run_id"]
        
        if run_id is not None:
            return run_id
        else:
            raise RuntimeError("No active MLflow run found.")