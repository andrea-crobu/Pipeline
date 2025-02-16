from .InputOutputPreprocessor import *

expected_input_preprocessing_params = {
    'tokenizer': None
}

expected_output_preprocessing_params = {
    'tokenizer': None
}

class TextSummarizationPreprocessor(InputOutputPreprocessor):
    def __init__(self, input_preprocessing_params, output_preprocessing_params):
        """
        Initialize the TextSummarizationPreprocessor with input and output preprocessing parameters.
        
        The prepocessing consists of tokenizing the input and output data.
        Then, the input is further processed in order to create a promt that improves the performance of the model. 
        
        Example:
                
            Here is a dialogue:

                {dialogue}
                
            Write a short summary!

        A AutoTokenizer.from_pretrained() tokenizer is expected a preprocessing parameter. It should be the same for both the input (dialogue) and the output (summarisation)
        
        Args:
            input_preprocessing_params (dict): Parameters specific to preprocessing the input data.
            output_preprocessing_params (dict): Parameters specific to preprocessing the output data.

        Raises:
            ValueError: If the input or output preprocessing parameters are invalid.
        """
        super().__init__(input_preprocessing_params, output_preprocessing_params)
        
    def format_prompt(self, text_dataframe):
        """
        Format a prompt from the input data.
        
        This method takes a pandas Series containing the dialogue and formats it into a prompt
        that is suitable for the model. The prompt should be a string that contains the dialogue
        and is formatted such that it will be correctly interpreted by the model.
        
        Args:
            text_dataframe (pd.Series): A pandas Series containing the dialogue to be formatted.
        
        Returns:
            dataframe: The formatted prompt.
        """

        prompt_start = 'Here is a dialogue:\n'
        prompt_end = '\n\nWrite a short summary!'
        
        formatted_prompt = text_dataframe.map(lambda x: prompt_start + x + prompt_end)
        return formatted_prompt
                        
    
    def preprocess_input_data(self, input_dataframe):
        """
        Preprocess the input data for text summarization.

        This method tokenizes the input data and formats it to create a prompt
        that enhances the model's performance. It expects a tokenizer to be provided
        in the input preprocessing parameters.

        Args:
            input_dataframe (pd.DataFrame): The input data containing dialogues to be preprocessed.

        Returns:
            tf.Tensor: The preprocessed input data ready for the model.
        """
        
        # first format the dialogues to match the desired prompt
        formatted_dialogues = self.format_prompt(input_dataframe)
        
        # tokenize the dialogues
        tokenizer = self.get_input_preprocessing_params()['tokenizer']
        tokenized_inputs = tokenizer(formatted_dialogues.to_list(), max_length=1024, truncation=True).input_ids
        
        return tokenized_inputs
    
    def preprocess_output_data(self, output_dataframe):
        """
        Preprocess the output data for text summarization.

        This method tokenizes the output data and formats it to create a prompt
        that enhances the model's performance. It expects a tokenizer to be provided
        in the output preprocessing parameters.

        Args:
            output_dataframe (pd.DataFrame): The output data containing dialogues to be preprocessed.

        Returns:
            tf.Tensor: The preprocessed output data ready for the model.
        """
        
        # tokenize the summarizations
        tokenizer = self.get_output_preprocessing_params()['tokenizer']
        tokenized_outputs = tokenizer(output_dataframe.to_list(), max_length=256, truncation=True).input_ids
        
        return tokenized_outputs
        
        