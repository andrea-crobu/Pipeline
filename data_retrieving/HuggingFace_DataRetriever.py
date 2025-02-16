from .DataRetriever import *

from datasets import load_dataset

class HuggingFace_DataRetriever(DataRetriever):
    def __init__(self, hugging_face_dataset_name:str):
        """
        Initializes a HuggingFace_DataRetriever instance.

        Args:
            hugging_face_dataset_name (str): The name of the Hugging Face dataset to retrieve.

        Returns:
            A HuggingFace_DataRetriever instance.
        """
        super().__init__()
        self.hugging_face_dataset_name = hugging_face_dataset_name
        
    def retrieve_data(self) -> dict:
        """
        Retrieve data from a Hugging Face dataset.

        Args:
            path (str): The path to the Hugging Face dataset.

        Returns:
            A dictionary containing the data from the Hugging Face dataset.
        """
        dataset = load_dataset(self.hugging_face_dataset_name)
        return dataset
        
        