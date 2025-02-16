from DataLoader import DataLoader

import tensorflow as tf
import numpy as np
from typing import Union, Optional, List  # Ensure proper type hinting imports

class LocalFile_DataLoader(DataLoader):
    """
    A specialized DataLoader for loading data from local NumPy files.
    
    This class extends the base DataLoader to work specifically with 
    local file paths instead of in-memory data. It expects data to be 
    saved as NumPy files (.npy) locally.
    
    Attributes:
        x (List): Paths to data files
        y (Optional[List]): Paths to label files (optional)
    """
    def __init__(
        self, 
        x: Union[np.ndarray, List],  # Input data file paths
        y: Optional[Union[np.ndarray, List]] = None,  # Optional label file paths
        batch_size: int = 32,  # Number of samples per batch
        shuffle: bool = True,  # Whether to randomize data order
        num_parallel_calls: int = tf.data.AUTOTUNE  # Parallel processing configuration
    ):
        # Call the parent class constructor with the provided parameters
        super().__init__(x, y, batch_size, shuffle, num_parallel_calls)
        
    def dataset_generator_function(self):
        """
        Generator function to load data from local files.
        
        Yields individual samples (and labels if available) by:
        1. Creating an index range of data files
        2. Optionally shuffling the indices
        3. Loading each file and yielding its contents
        4. Handling potential file loading errors gracefully
        """
        # Create an array of indices corresponding to file paths
        indices = np.arange(len(self._x))
        
        # Randomly shuffle indices if shuffle is enabled
        if self._shuffle:
            # Ensures batches are created from randomized data order
            np.random.shuffle(indices)
        
        # Iterate through shuffled or original indices
        for idx in indices:
            try:
                # Load data sample from file path
                sample = np.load(self._x[idx])
                
                # If labels are provided, load corresponding label
                if self._y is not None:
                    label = np.load(self._y[idx])
                    yield sample, label
                else:
                    # Yield only the sample if no labels are present
                    yield sample

            # Comprehensive error handling for file loading
            except FileNotFoundError:
                # Log and skip files that cannot be found
                print(f"File not found")
                continue
            except PermissionError:
                # Log and skip files with permission issues
                print(f"Permission denied for file")
                continue
            except Exception as e:
                # Catch and log any unexpected errors during file loading
                print(f"Unexpected error loading file: {e}")
                continue

    def get_data_dimensions(self):
        """
        Retrieve the shape of the first data sample.
        
        Returns:
            Numpy array shape of the first data file
        """
        # Load and return the shape of the first data file
        return np.load(self._x[0]).shape

    def get_label_dimensions(self):
        """
        Retrieve the shape of the first label sample.
        
        Returns:
            Numpy array shape of the first label file, or None if no labels
        """
        # Return label shape if labels exist, otherwise return None
        return np.load(self._y[0]).shape if self._y is not None else None