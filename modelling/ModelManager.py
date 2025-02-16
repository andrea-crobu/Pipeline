import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

from dataloader.DataLoader import DataLoader
from training.TrainingManager import TrainingManager
from evaluation.EvaluationManager import EvaluationManager
from logging.MLFlow_LoggingManager import  MLFlow_LoggingManager
from ..preprocessing.InputOutputPreprocessor import InputOutputPreprocessor

class ModelManager:
    """
    Orchestrates the model training, evaluation, inference, and logging.
    Delegates specific tasks to specialized managers for scalability and readability.
    """

    def __init__(self, 
                 model: tf.keras.Model, 
                 training_set_dataloader: DataLoader, 
                 validation_set_dataloader: DataLoader, 
                 test_set_dataloader: DataLoader,
                 training_parameters: dict,
                 evaluation_parameters: dict,  
                 mlflow_logging_parameters: dict,
                 input_output_preprocessor: InputOutputPreprocessor):
        """
        Initialize the ModelManager to manage training, evaluation, inference, and logging.
        """
        # Store the model and post-process layer
        self._model = model

        # Initializ e the specialized managers
        self._training_manager = TrainingManager(model, training_parameters, training_set_dataloader, validation_set_dataloader)
        self._evaluation_manager = EvaluationManager(model, evaluation_parameters, test_set_dataloader)
        self._logging_manager = MLFlow_LoggingManager(mlflow_logging_parameters)

        # Store the parameters
        self._training_parameters = training_parameters
        self._evaluation_parameters = evaluation_parameters

        # Store the dataloaders
        self._training_set_dataloader = training_set_dataloader
        self._validation_set_dataloader = validation_set_dataloader
        self._test_set_dataloader = test_set_dataloader

        # Store the preprocessor
        self._input_output_preprocessor = input_output_preprocessor

    # Main methods that orchestrate the workflow
    
    # Training methods
    def train_and_log_results(self):
        """
        Trains the model using the given parameters and logs the results.
        """
        self._logging_manager.start_logging()
        self._logging_manager.log_parameters()
        
        try:
            self._training_manager.train()
            self._logging_manager.log_model(self._model)
        finally:
            self._logging_manager.end_logging()
            
    def train_with_validation_and_log_results(self):
        """
        Trains the model using validation data for periodic evaluation.
        """
        self._logging_manager.start_logging()
        self._logging_manager.log_parameters()
        
        try:
            self._training_manager.train_with_validation()
            self._logging_manager.log_model(self._model)
        finally:
            self._logging_manager.end_logging()

    def train_with_cross_validation_and_log_results(self):
        """
        Performs cross-validation to train the model across multiple data splits.
        
        Notes:
        - Logs model and performance for each fold
        - Ensures logging is stopped even if training fails
        """
        
        # Get total dataset size
        idx_of_all_data = np.arange(len(self._training_set_dataloader.get_x()))
        
        # Validate k-fold parameter
        n_splits = self._training_parameters.get('k_fold')
        if n_splits is None or n_splits < 2:
            raise ValueError("Invalid k_fold. Must be an integer >= 2")
        
        # Perform k-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(idx_of_all_data), start=1):
            
            self._logging_manager.start_logging()
            self._logging_manager.log_parameters()
            
            try:    
                print(f"Training Fold {fold}/{n_splits}")
                
                # Create fold-specific dataloaders
                train_fold_dataloader = type(self._training_set_dataloader)(
                    self._training_set_dataloader.get_x()[train_idx],
                    self._training_set_dataloader.get_y()[train_idx], 
                    batch_size=self._training_set_dataloader.get_batch_size(), 
                    shuffle=self._training_set_dataloader.get_shuffle()
                )
                
                val_fold_dataloader = type(self._validation_set_dataloader)(
                    self._validation_set_dataloader.get_x()[val_idx],
                    self._validation_set_dataloader.get_y()[val_idx], 
                    batch_size=self._validation_set_dataloader.get_batch_size(), 
                    shuffle=self._validation_set_dataloader.get_shuffle()
                )
                                
                # Train on current fold
                self._training_manager.train_with_cross_validation(train_fold_dataloader, val_fold_dataloader)
                                
            except Exception as e:
                print(f"Cross-validation error: {e}")
                raise
            
            finally:
                self._logging_manager.end_logging()
        
    # Evaluation methods
    def evaluate_model_and_log_results(self):
        """
        Evaluates the model and logs the results.
        """
        self._logging_manager.start_logging(self._logging_manager.get_active_run_id())
        
        try:
            evaluation_results = self._evaluation_manager.evaluate_model()
            self._logging_manager.log_evaluation_metrics(evaluation_results)
        finally:
            self._logging_manager.end_logging()

    # Inference methods    
    def run_inference_on_dataloader(self, dataloader: DataLoader, inference_parameters: dict)-> np.ndarray:
        """
        Run the model on arbitrary input data for inference. 
        
        Args:
            dataloader (DataLoader): DataLoader containing input data for inference.
        
        Returns:
            np.ndarray: Predictions generated by the model.
        """
        batch_size = inference_parameters['batch_size'] if inference_parameters['batch_size'] is not None else None
        predictions = self._model.predict(dataloader, batch_size)
        
        return predictions
    
    def run_inference_on_raw_data(self, input_data: np.ndarray, inference_parameters: dict) -> np.ndarray:
        """
        Run the model on raw input data for inference.
        
        Args:
            input_data (np.ndarray): Raw input data for inference.
            inference_parameters (dict): Inference parameters.
        Returns:
            np.ndarray: Predictions generated by the model.
        """
        # Preprocess the input data
        input_data = self._input_output_preprocessor.preprocess_input_data(input_data)
        
        # Create a DataLoader for the input data
        input_dataloader = type(self._test_set_dataloader)(
            input_data, 
            batch_size=self._test_set_dataloader.get_batch_size(), 
            shuffle=self._test_set_dataloader.get_shuffle()
        )
        
        return self.run_inference_on_dataloader(input_dataloader, inference_parameters)
    
    # Getters for private attributes
    def get_model(self):
        return self._model

    def get_post_process_layer(self):
        return self._post_process_layer

    def get_training_set_dataloader(self):
        return self._training_set_dataloader

    def get_validation_set_dataloader(self):
        return self._validation_set_dataloader

    def get_test_set_dataloader(self):
        return self._test_set_dataloader








