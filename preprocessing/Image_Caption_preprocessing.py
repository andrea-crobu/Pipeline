from .InputOutputPreprocessor import InputOutputPreprocessor

from nltk.tokenize import RegexpTokenizer
import re

from transformers import AutoTokenizer

import pandas as pd

expected_input_preprocessing_params = {} # no preprocessing at this level, it will be handed by the dataloader

expected_output_preprocessing_params = {
    'lower_case':True,
    'remove_punctuation':True,
    'remove_stopwords':True,
    'remove_digits':True,
    'tokenizer':None
    }

class Image_Caption_preprocessing(InputOutputPreprocessor):
    def __init__(self, input_preprocessing_params, output_preprocessing_params):
        super().__init__(input_preprocessing_params, output_preprocessing_params)
        
    def preprocess_input_data(self, input_dataframe):
        pass
    
    def preprocess_output_data(self, captions: pd.Series) -> list:
        
        # Apply lowercase
        captions = captions.map(lambda s: str.lower(s.strip()))  # Strip leading/trailing spaces

        # Remove numbers, special characters, and punctuation
        captions = captions.map(lambda caption: re.sub(r'[^a-zA-Z\s]+', '', caption))  
        captions = captions.map(lambda caption: re.sub(r'\s+', ' ', caption).strip())  # Normalize spaces

        # now tokenize each caption
        # create tokenizer
        tk = RegexpTokenizer(r'\w+')
        captions = captions.map(lambda caption: tk.tokenize(caption))

        # add the token <sos> at the beginning of each caption and the <eos> at the end of each caption
        captions = captions.map(lambda caption: ['<sos>'] + caption + ['<eos>'])
        
        # use the desired tokenizer for the captions
        tokenizer = self.get_output_preprocessing_params()['tokenizer'](captions.to_list())
        self.get_output_preprocessing_params()['tokenizer'] = tokenizer # save the tokenizer for later
        
        captions = captions.map(lambda caption: tokenizer(caption))      
        
        # create the list
        list_of_captions = captions.to_list()
        
        return list_of_captions
    

    
    
    