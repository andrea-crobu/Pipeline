from .InputOutputPreprocessor import *

# The expected input and output preprocessing parameters must be defined in the application 
expected_input_preprocessing_params = {
    'tokenizer': None,
    'max_length': None,
    'truncation': None
}
expected_output_preprocessing_params = {
    'label_preprocess_fn': None
}

class SentimentAnalysisPreprocessor(InputOutputPreprocessor):
    def __init__(self, input_preprocessing_params, output_preprocessing_params):
        super().__init__(input_preprocessing_params, output_preprocessing_params)
        
    
    def preprocess_input_data(self, input_dataframe):
        """
        Preprocess the input data for sentiment analysis.

        This method tokenizes the input data. It expects a tokenizer to be provided
        in the input preprocessing parameters.

        Args:
            input_dataframe (pd.DataFrame): The input data containing text to be preprocessed.

        Returns:
            tokenized_inputs (list): The preprocessed input data.
        """
        tokenizer = self.get_input_preprocessing_params()['tokenizer']
        max_length = self.get_input_preprocessing_params()['max_length']
        truncation = self.get_input_preprocessing_params()['truncation']
        
        tokenized_inputs = input_dataframe.apply(lambda x: tokenizer(str(x), max_length=max_length, truncation=truncation).input_ids).to_list()
        
        return tokenized_inputs
    
    def preprocess_output_data(self, ouput):
        """
        Preprocess the output data for sentiment analysis.

        This method applies a label preprocessing function provided in the output preprocessing parameters.

        Args:
            ouput (list): The output data containing sentiment labels to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed output data.
        """
        label_preprocess_fn = self.get_output_preprocessing_params()['label_preprocess_fn']
        labels = ouput.apply(label_preprocess_fn).to_list()
        
        return labels
    
    