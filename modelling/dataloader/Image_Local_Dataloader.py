from .DataLoader import *
from PIL import Image

class Image_Local_DataLoader(DataLoader):
    def __init__(
        self, 
        x: Union[np.ndarray, List],
        y: Optional[Union[np.ndarray, List]] = None, 
        batch_size: int = 32, 
        shuffle: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        image_preprocessing_fn: Optional[Callable] = None,
        fixed_image_size = None
    ):
        
        # Store the image preprocessing function
        self._image_preprocessing_fn = image_preprocessing_fn   
        
        # Call the parent class constructor
        super().__init__(x, y, batch_size, shuffle, num_parallel_calls)
        
        # Store the fixed image size
        self._fixed_image_size = fixed_image_size     
        
        
    def dataset_generator_function(self):
        """
        Generator function to load image data from local.
        
        Yields individual samples by:
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
                # Load image sample from file path
                sample = Image.open(self._x[idx])
                
                # apply the preprocessing expected by the model
                if self._image_preprocessing_fn is not None:
                    sample = self._image_preprocessing_fn(sample)
                
                # Yield only the sample
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
        Retrieve the shape of the first image sample.
        
        Returns:
            Numpy array shape of the first data file
        """
        # Load and return the shape of the first data file
        image_example = Image.open(self._x[0])
        preprocessed_image = self._image_preprocessing_fn(image_example)
        expected_shape = preprocessed_image.shape
        return expected_shape
    
    
        
        
        
        
        
        