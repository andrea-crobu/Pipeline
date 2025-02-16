import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from natsort import natsorted
import shutil

class DataRetriever:
    """
    The DataRetriever class is an abstract base class designed for retrieving data from various sources.
    
    This class serves as a template that can be extended to define specific data retrieval mechanisms 
    based on the source and format of the data. It includes methods to generate a list of files based 
    on a query, retrieve data from those files, and store the data in a dataframe.
    """

    def __init__(self):
        """
        Initializes an instance of the DataRetriever class.
        
        Attributes:
            _dataframe: A private attribute that holds the retrieved data as a dataframe.
        """
        self._dataframe = None  # Placeholder for the dataframe that will store retrieved data
    
    def generate_list_of_files(self, query) -> list:
        """
        Generate a list of files based on a query.
        
        This is an abstract method, intended to be implemented by subclasses that define the
        specific logic for generating a list of files based on a given query (e.g., searching
        a directory or querying a database).
        
        Args:
            query: The query used to generate the list of files (e.g., search parameters, directory path).
        
        Returns:
            list_of_files (list): A list of files that match the query. 
        
        Raises:
            NotImplementedError: Since this method is abstract, it raises an error if not implemented by a subclass.
        """
        pass
    
    def retrieve_data(self, list_of_files: list):
        """
        Retrieve data from a list of files and store it in a dataframe.
        
        This is an abstract method, intended to be implemented by subclasses to define the logic
        for retrieving data from a list of files (e.g., reading CSV files into a dataframe).
        
        Args:
            list_of_files (list): A list of files from which data will be retrieved.
        
        Raises:
            NotImplementedError: Since this method is abstract, it raises an error if not implemented by a subclass.
        """
        self._dataframe = None  # Initialize the dataframe as None
        raise NotImplementedError  # To be implemented by subclass
    
    def save_data_locally(self, dataframe, file_path: str, delete_existing:bool=False):
        """
        Save the data contained in the dataframe locally.
        """ 
        
        # if data already exist, delete it (if delete_existing=True)
        if delete_existing and os.path.exists(file_path):
            shutil.rmtree(file_path)
        
        # Create the dataset directory if it doesn't exist
        os.makedirs(file_path, exist_ok=True)
        
        # Set an index based on the number of files already in the directory
        if len(os.listdir(file_path)) > 0:
            index = len(os.listdir(file_path)) + 1
        else:
            index = 1
        
        # Convert the DataFrame to a PyArrow table
        table = pa.Table.from_pandas(dataframe)

        # Write the new table to the dataset directory
        pq.write_to_dataset(
            table,
            root_path=file_path+f'/chunk_{index}',
            )
        
    def load_data_locally(self, file_path: str):
        """
        Load the data from a local file and store it in a dataframe.
        """
        
        # Get all Parquet files in the directory
        files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
        
        # Sort files by name
        files = natsorted(files)

        # Load the data in order
        tables = [pq.read_table(file) for file in files]

        # Combine all tables
        combined_table = pa.concat_tables(tables)


        # combined_table = pq.read_table(file_path)
        
        dataframe = combined_table.to_pandas()

        return dataframe
    
    # Getter methods
    def get_data(self):
        """
        Get the retrieved data as a dataframe.
        
        Returns:
            dataframe (pandas.DataFrame): The dataframe containing the retrieved data.
        
        Notes:
            The returned dataframe may be empty if the data retrieval was unsuccessful.
        """
        return self._dataframe  # Return the dataframe containing the retrieved data
