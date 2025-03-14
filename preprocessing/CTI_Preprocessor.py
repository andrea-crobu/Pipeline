from .InputOutputPreprocessor import *

import ast
import pandas as pd

# Define expected parameters for input and output preprocessing
expected_input_preprocessing_params = {
    'tokenizer': None,
    'truncation': None
}

expected_output_preprocessing_params = {
    'tokenizer': None,
    'truncation': None
}

class CTI_Preprocessor(InputOutputPreprocessor):
    """
    Cyber Threat Intelligence (CTI) Preprocessor for formatting and tokenizing threat data.
    
    This preprocessor handles the transformation of threat reports into structured prompts
    and processes the extracted entities and relations.
    
    Required DataFrame columns:
    - For input preprocessing: 'text' (contains the threat report content)
    - For output preprocessing: 'text', 'entities', 'relations', 'diagnosis'
        - 'text': Original threat report text
        - 'entities': Representation of entity objects list
        - 'relations': Representation of relation objects list
        - 'diagnosis': Analysis of the threat
    """
    
    def __init__(self, input_preprocessing_params, output_preprocessing_params):
        """
        Initialize the CTI Preprocessor with tokenizers and processing parameters.
        
        Args:
            input_preprocessing_params (dict): Parameters for input preprocessing
            output_preprocessing_params (dict): Parameters for output preprocessing
        """
        super().__init__(input_preprocessing_params, output_preprocessing_params)
    
    def format_input_prompt(self,dataframe):
        
        
        
        # create the desired prompt
        first_part = f"You are a skilled AI Agent capable of doing CTI Analysis.\n\nGiven this threat report:\n"
        second_part = "\n\nYou will extract the main entities and their relations; finally, you will generate a diagnosis of the threat.\n"
            
        formatted_input_prompt = dataframe.apply(lambda x: pd.Series(
            {
                'prompt': f"{first_part}{x['text']}{second_part}"
            }),
            axis=1
            )
            
        return formatted_input_prompt
    
    def process_entries_and_relations(self, dataframe_row):
        """
        Process entities and relations from a dataframe row.
        
        Args:
            dataframe_row (pandas.Series): A row containing 'text', 'entities', and 'relations'
            
        Returns:
            pandas.Series: The processed row with formatted entities and relations
            
        Note:
            Required columns in dataframe_row:
            - 'text': The original threat report text
            - 'entities': String representation of a list of entity dictionaries
            - 'relations': String representation of a list of relation dictionaries
        """
                    
        # Make a copy to avoid modifying the original row
        row = dataframe_row.copy()
        
        # Convert string representations to Python objects (dict in this case)
        row['entities'] = ast.literal_eval(row['entities']) if isinstance(row['entities'], str) else row['entities']
        row['relations'] = ast.literal_eval(row['relations']) if isinstance(row['relations'], str) else row['relations']
        
        # Create a dictionary to store entities by id for faster lookups
        entities_dict = {}
        entities_formatted = []
        
        # Process entities
        for entity in row['entities']:
            token = row['text'][entity['start_offset']:entity['end_offset']]
            label = entity['label']
            entity_id = entity['id']
            
            # Store in dictionary for quick lookup when processing relations
            entities_dict[entity_id] = {
                'entity_name': token,
                'label': label
            }
            
            # Format entity for display
            entities_formatted.append(f'{token} ({label})')
        
        # Process relations with faster lookups
        relations_formatted = []
        
        for relation in row['relations']:
            from_id = relation['from_id']
            to_id = relation['to_id']
            relation_type = relation['type']
            
            # Use the dictionary for lookups instead of filtering the DataFrame
            if from_id in entities_dict and to_id in entities_dict:
                entity1 = entities_dict[from_id]['entity_name']
                entity2 = entities_dict[to_id]['entity_name']
                relations_formatted.append(f'{entity1} to {entity2} ({relation_type})')
        
        # Update the row with formatted strings
        row['entities'] = ', '.join(entities_formatted)
        row['relations'] = ', '.join(relations_formatted)
        
        return row

    def format_output(self, dataframe):
        first_part = f"\nEntities: "
        second_part = f"\nRelations: "
        third_part = f"\nDiagnosis: "
        
        formatted_output = dataframe.apply(lambda x: pd.Series(
            {
                'output': f"{first_part}{x['entities']}{second_part}{x['relations']}{third_part}{x['diagnosis']}"
            }
        ),
        axis=1
        )
        
        return formatted_output

    def preprocess_input_data(self, input_dataframe, for_training:bool = True):
        """
        Preprocess input data for model training or inference.
        
        Args:
            input_dataframe (pandas.DataFrame): DataFrame with threat report data
                                              Required column: 'text'
                                              
        Returns:
            list: Tokenized inputs ready for model
        """
        # Validate input dataframe has required column
        required_cols = ['text', 'entities', 'relations', 'diagnosis']
        missing_cols = [col for col in required_cols if col not in input_dataframe.columns]
        if missing_cols:
            raise ValueError(f"Output dataframe missing required columns: {', '.join(missing_cols)}")
        
        # Process entities and relations
        input_dataframe = input_dataframe.apply(lambda x: self.process_entries_and_relations(x), axis=1)
        
        # Now, let's create the desired prompt
        formatted_prompt = self.format_input_prompt(input_dataframe)
        
        if for_training:
            # Now let's format the output
            formatted_output = self.format_output(input_dataframe)
            
            # Tokenize the input prompt and the output
            tokenizer = self.get_input_preprocessing_params()['tokenizer']      
            truncation = self.get_input_preprocessing_params()['truncation'] 
            tokenized_input_prompt = formatted_prompt.apply(lambda x: tokenizer(x['prompt'],truncation=truncation).input_ids, axis=1) # pd.Series()
            
            tokenizer = self.get_output_preprocessing_params()['tokenizer']      
            truncation = self.get_output_preprocessing_params()['truncation'] 
            
            tokenized_output = formatted_output.apply(lambda x: tokenizer(x['output'], truncation=truncation).input_ids, axis=1) # pd.Series()
            
            # Now let's combine the prompt and the output, so that the model will learn to generate the output from the prompt
            tokenized_input_prompt = tokenized_input_prompt + tokenized_output 
            
            # add the eos_token
            eos_token = tokenizer(tokenizer.eos_token).input_ids
            tokenized_input_prompt = tokenized_input_prompt.apply(lambda x: x + eos_token)
        
        else:
            # Tokenize the input prompt
            tokenizer = self.get_input_preprocessing_params()['tokenizer']      
            truncation = self.get_input_preprocessing_params()['truncation'] 
            tokenized_input_prompt = formatted_prompt.apply(lambda x: tokenizer(x['prompt'], truncation=truncation).input_ids, axis=1) # pd.Series()
            
            # do not add the eos_token
            # eos_token = tokenizer(tokenizer.eos_token).input_ids
            # tokenized_input_prompt = tokenized_input_prompt.apply(lambda x: x + eos_token)
        
        return tokenized_input_prompt.to_list()
    
    def preprocess_output_data(self, output_dataframe):

        # Validate output dataframe has required columns
        required_cols = ['text', 'entities', 'relations', 'diagnosis']
        missing_cols = [col for col in required_cols if col not in output_dataframe.columns]
        if missing_cols:
            raise ValueError(f"Output dataframe missing required columns: {', '.join(missing_cols)}")
        
        # Now let's create the desired output: this includes the input (masked) and the desired output that the model should learn.
        
        # Process entities and relations
        output_dataframe = output_dataframe.apply(lambda x: self.process_entries_and_relations(x), axis=1)
        
        # generate the prompt, as we did for the input. This will be masked after the tokenization
        formatted_prompt = self.format_input_prompt(output_dataframe) 
        
        # generate the expected output
        formatted_outputs = self.format_output(output_dataframe)
        
        # Tokenize
        tokenizer = self.get_output_preprocessing_params()['tokenizer']      
        truncation = self.get_output_preprocessing_params()['truncation'] 
        
        tokenized_outputs = formatted_outputs.apply(lambda x: tokenizer(x['output'], truncation=truncation).input_ids, axis=1) # pd.Series()
        
        # Create the masks for the prompt, to avoid training on it (-100 is ignored in loss computation).
        # The mask is a list of -100, equal to the length of each tokenized prompt
        tokenizer = self.get_input_preprocessing_params()['tokenizer']      
        truncation = self.get_input_preprocessing_params()['truncation'] 
        tokenized_input_prompt = formatted_prompt.apply(lambda x: tokenizer(x['prompt'],truncation=truncation).input_ids, axis=1) # pd.Series()
        
        masks = tokenized_input_prompt.apply(len).apply(lambda x: x*[-100])
        
        # combine the mask to the outputs
        tokenized_outputs = masks + tokenized_outputs
        
        # add the eos_token
        eos_token = tokenizer(tokenizer.eos_token).input_ids
        tokenized_outputs = tokenized_outputs.apply(lambda x: x + eos_token)
                    
        return tokenized_outputs.to_list()