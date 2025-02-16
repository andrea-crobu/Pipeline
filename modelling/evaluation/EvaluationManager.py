import tensorflow as tf
from typing import Dict, Any, Optional, List, Union

from ..dataloader.DataLoader import DataLoader

class EvaluationManager:
    """
    Manages the evaluation process of machine learning models.
    
    This class provides a structured approach to evaluating model performance
    on test datasets, encapsulating the evaluation logic and providing 
    convenient access to the model.
    
    Attributes:
        _model (tf.keras.Model): The trained machine learning model to be evaluated
        _test_set_dataloader (DataLoader): DataLoader containing the test dataset
    """

    def __init__(
        self,
        model: tf.keras.Model,
        test_set_dataloader: DataLoader,
    ):
        """
        Initialize the EvaluationManager with a model and test dataset.
        
        Args:
            model: The trained TensorFlow/Keras model to be evaluated
            test_set_dataloader: DataLoader containing the test dataset
        
        Note:
            The model should be pre-trained before evaluation
        """
        # Store the input model 
        self._model: tf.keras.Model = model
        
        # Store the test set dataloader
        self._test_set_dataloader: DataLoader = test_set_dataloader
        
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Executes the model evaluation on the test dataset.
        
        Evaluates the model's performance using the test set dataloader.
        
        Returns:
            A dictionary containing evaluation metrics 
            (e.g., loss, accuracy, custom metrics defined during model compilation)
        
        Example return value might look like:
            {
                'loss': 0.123,
                'accuracy': 0.95,
                'precision': 0.92,
                'recall': 0.88
            }
        """
        # Use Keras evaluate method to compute model performance        
        return self._model.evaluate(self._test_set_dataloader, return_dict=True) # return_dict=True ensures metrics are returned as a dictionary

    def get_model(self) -> tf.keras.Model:
        """
        Retrieve the model used for evaluation.
        
        Returns:
            The TensorFlow/Keras model
        """
        return self._model