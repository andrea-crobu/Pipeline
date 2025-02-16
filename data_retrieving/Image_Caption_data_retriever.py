from .DataRetriever import DataRetriever
import pandas as pd

class Image_Caption_data_retriever(DataRetriever):
    def __init__(self):
        super().__init__()
        
    def retrieve_data(self, path:str):
        self._dataframe = pd.read_csv(path, sep='\t', names=['image_ID','caption'])
        
        # create the list of image IDs
        list_of_image_IDs = []

        list_of_image_IDs = self._dataframe['image_ID'].to_list()

        # the image name contains an ending that must be removed (#1, #2, #3, #4, #5)
        clean_list_of_image_IDs = []
        for image_ID in list_of_image_IDs:
            clean_list_of_image_IDs.append(image_ID.split('#')[0])

        # overwrite the image ID list
        list_of_image_IDs = clean_list_of_image_IDs

        # overwrite the image_ID columns in the dataframe
        self._dataframe['image_ID'] = pd.DataFrame(list_of_image_IDs)
        
            
    
    
    