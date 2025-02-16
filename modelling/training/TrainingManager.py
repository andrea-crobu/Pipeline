import tensorflow as tf
from typing import Dict, Any, Optional, List, Union

from ..dataloader.DataLoader import DataLoader

# Expected training parameters
expected_training_params: Dict[str, Optional[Any]] = {
    'loss': None,           # Can be a string, callable, or None
    'optimizer': None,      # TensorFlow optimizer
    'metrics': None,        # List of metrics or None
    'batch_size': None,     # Integer batch size
    'epochs': None,         # Integer number of epochs
    'callbacks': None       # List of Keras callbacks or None
}

class TrainingManager:
    def __init__(
        self, 
        model: tf.keras.Model, 
        training_parameters: Dict[str, Any], 
        training_set_dataloader: DataLoader, 
        validation_set_dataloader: Optional[DataLoader] = None
    ):
        """
        Initialize the TrainingManager with model and training configuration.
        
        Args:
            model: The TensorFlow/Keras model to be trained
            training_parameters: Dictionary containing training configuration
            training_set_dataloader: DataLoader for training data
            validation_set_dataloader: Optional DataLoader for validation data
        """
        # Store the input model and parameters with type-hinted attributes
        self._model: tf.keras.Model = model
        self._training_parameters: Dict[str, Any] = training_parameters
        self._training_set_dataloader: DataLoader = training_set_dataloader
        self._validation_set_dataloader: Optional[DataLoader] = validation_set_dataloader
        
        # Validate the provided training parameters
        self._validate_training_parameters()
        
        # Prepare the model for training
        self._model = self._setup_model(self._model)        

    def _validate_training_parameters(self) -> None:
        """
        Validates that all required training parameters are present and not None.
        
        Raises:
            ValueError: If a required parameter is missing or set to None
        """
        # Check each expected parameter
        for key in expected_training_params.keys():
            # Ensure the key exists in the training parameters
            if key not in self._training_parameters.keys():
                raise ValueError(f"Missing key {key} in training parameters.")
            
            # Ensure the parameter value is not None
            elif self._training_parameters[key] is None:
                raise ValueError(f"Value for key {key} in training parameters is None.")
            
    def _setup_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Compiles the model with the specified loss, optimizer, and metrics.
        """
        # Compile the model using the provided training parameters
        model.compile(
            loss=self._training_parameters['loss'],
            optimizer=self._training_parameters['optimizer'],
            metrics=self._training_parameters['metrics']
        )
        
        # Print model summary to console
        model.summary()
        
        # TODO: Remove this after debugging - this makes training slower
        model.run_eagerly = True 
        
        return model
    
    def train(self) -> None:
        """
        Executes the standard training loop for the model without validation.
        
        Returns:
            Training history object containing metrics
        """
        return self._model.fit(
            x=self._training_set_dataloader,
            epochs=self._training_parameters['epochs'],
            callbacks=self._training_parameters['callbacks']
        )

    def train_with_validation(self) -> None:
        """
        Executes the training loop with periodic validation.
        
        Returns:
            Training history object containing metrics
        """
        # Ensure validation dataloader is not None
        if self._validation_set_dataloader is None:
            raise ValueError("Validation dataloader is not provided")
        
        return self._model.fit(
            x=self._training_set_dataloader,
            validation_data=self._validation_set_dataloader,
            epochs=self._training_parameters['epochs'],
            callbacks=self._training_parameters['callbacks']
        )
    
    def train_with_cross_validation(
        self, 
        fold_training_set_dataloader: DataLoader, 
        fold_validation_set_dataloader: DataLoader
    ) -> None:
        """
        Executes the training loop for a specific fold in cross-validation.
        
        Args:
            fold_training_set_dataloader: DataLoader for the current fold's training data
            fold_validation_set_dataloader: DataLoader for the current fold's validation data
        
        Returns:
            Training history object containing metrics
        """
        # Clear the current TensorFlow session to free up resources
        tf.keras.backend.clear_session()
        
        # Create a clone of the original model to avoid modifying the base model
        copied_model = tf.keras.models.clone_model(self._model)
        
        # Recompile the cloned model
        copied_model = self._setup_model(copied_model)        
        
        # Train on the current fold's data
        return copied_model.fit(
            x=fold_training_set_dataloader,
            validation_data=fold_validation_set_dataloader,
            epochs=self._training_parameters['epochs'],
            callbacks=self._training_parameters['callbacks']
        )
    
    # Getter methods to access internal properties
    def get_model(self) -> tf.keras.Model:
        """
        Returns the current model.
        
        Returns:
            The TensorFlow/Keras model
        """
        return self._model

    def get_training_set_dataloader(self) -> DataLoader:
        """
        Returns the training set dataloader.
        
        Returns:
            The DataLoader for training data
        """
        return self._training_set_dataloader
    
    def get_validation_set_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the validation set dataloader.
        
        Returns:
            The DataLoader for validation data, which may be None
        """
        return self._validation_set_dataloader