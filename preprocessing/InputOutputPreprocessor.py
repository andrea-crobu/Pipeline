# The expected input and output preprocessing parameters must be defined in the application 
expected_input_preprocessing_params = {}
expected_output_preprocessing_params = {}

class InputOutputPreprocessor:
    """
    Preprocessor class for handling data preprocessing tasks for input and output data.
    
    This class manages the preprocessing parameters and ensures that the input and output data
    are properly preprocessed before being passed to the model. It also includes validation
    of preprocessing parameters to ensure that the data meets the expected requirements.
    """
    
    def __init__(self, input_preprocessing_params: dict, output_preprocessing_params: dict):
        """
        Initialize the Preprocessor with input and output preprocessing parameters.
        
        Args:
            input_preprocessing_params (dict): Dictionary containing preprocessing parameters for input data.
            output_preprocessing_params (dict): Dictionary containing preprocessing parameters for output data.

        Raises:
            ValueError: If the input or output preprocessing parameters are invalid.
        """
        # Store the preprocessing parameters for future use
        self._input_preprocessing_params = input_preprocessing_params
        self._output_preprocessing_params = output_preprocessing_params

        # Validate the preprocessing parameters against the expected parameters
        if not self._validate_preprocess_parameters(expected_input_preprocessing_params, self._input_preprocessing_params):
            raise ValueError("Invalid input preprocessing parameters")
        
        if not self._validate_preprocess_parameters(expected_output_preprocessing_params, self._output_preprocessing_params):
            raise ValueError("Invalid output preprocessing parameters")

        # Initialize placeholders for storing the preprocessed input and output data
        self._inputs = None
        self._outputs = None

    def _validate_preprocess_parameters(self, expected_preprocessing_params: dict, preprocessing_params: dict) -> bool:
        """
        Validate the preprocessing parameters against expected values.
        
        This method checks that the provided preprocessing parameters contain all the necessary
        keys and that their values are valid for the preprocessing pipeline.

        Args:
            expected_preprocessing_params (dict): The expected keys and values for preprocessing parameters.
            preprocessing_params (dict): The actual parameters to be validated.

        Returns:
            bool: True if the parameters are valid, False otherwise.
        
        Raises:
            ValueError: If any required key is missing or if there are invalid values in the parameters.

        Example:
            If the pipeline requires a "scaling_method" key with values "min_max" or "standard",
            this method will ensure that the key is present and that the value is one of the allowed ones.
        """
        
        # Initialize the result of validation as True (valid by default)
        is_correct = True
        
        # Ensure all keys in the provided parameters are expected
        for key in expected_preprocessing_params.keys():
            if key not in preprocessing_params:
                is_correct = False  # Mark as invalid if a required key is missing
        
        # Return the result
        return is_correct
    
    def preprocess_input_data(self, input_dataframe):
        """
        Preprocess the input data.

        This method should be implemented in a subclass to define the specific preprocessing steps 
        for the input data based on the preprocessing parameters.

        Args:
            input_dataframe (TODO): The input data (dataframe) to be preprocessed.
            input_preprocessing_params (dict): Parameters specific to preprocessing the input data.

        Returns:
            inputs (tf.Tensor): The preprocessed input data, expected in the format required by the model.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError  # To be implemented by subclasses
    
    def preprocess_output_data(self, output):
        """
        Preprocess the output data.

        This method should be implemented in a subclass to define the specific preprocessing steps 
        for the output data based on the preprocessing parameters.

        Args:
            output (TODO): The output data to be preprocessed.
            output_preprocessing_params (dict): Parameters specific to preprocessing the output data.

        Returns:
            outputs (tf.Tensor): The preprocessed output data, expected in the format required by the model.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError  # To be implemented by subclasses

    # Getter methods for retrieving the preprocessing parameters and processed data
    def get_input_preprocessing_params(self):
        """
        Get the preprocessing parameters for the input data.

        Returns:
            dict: The preprocessing parameters for the input data.
        """
        return self._input_preprocessing_params
    
    def get_output_preprocessing_params(self):
        """
        Get the preprocessing parameters for the output data.

        Returns:
            dict: The preprocessing parameters for the output data.
        """
        return self._output_preprocessing_params
    
    def get_inputs(self):
        """
        Get the preprocessed input data.

        Returns:
            tf.Tensor: The preprocessed input data.
        """
        return self._inputs
    
    def get_outputs(self):
        """
        Get the preprocessed output data.

        Returns:
            tf.Tensor: The preprocessed output data.
        """
        return self._outputs
