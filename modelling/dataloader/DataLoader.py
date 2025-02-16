
import tensorflow as tf
import numpy as np
from typing import Union, Optional, List, Tuple, Callable


class DataLoader:
    def __init__(
        self, 
        x: Union[np.ndarray, List], 
        y: Optional[Union[np.ndarray, List]] = None,
        batch_size: int = 32, 
        shuffle: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE):
        """
        Flexible TensorFlow DataLoader using tf.data.Dataset.from_generator()
        
        Args:
            x: Input features (NumPy array or list)
            y: Optional corresponding labels (NumPy array or list)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            extra_preprocessing_fn: Optional preprocessing function for data
            num_parallel_calls: Parallelization strategy for data loading
        
        Note: 
        - Supports both labeled and unlabeled datasets
        - Utilizes TensorFlow's efficient data loading pipeline
        """
        
        # Convert inputs to NumPy arrays for consistency
        self._x = np.asarray(x)                           
        self._y = np.asarray(y,dtype=object) if y is not None else None                           
        
        # Validate input dimensions
        if self._y is not None and len(self._x) != len(self._y):
            raise ValueError(f"Input features and labels must have the same length. \n x: {len(self._x)} and y: {len(self._y)} ")
        
        self._batch_size = batch_size          
        self._shuffle = shuffle
        self._num_parallel_calls = num_parallel_calls
        
        self._dataset = self.create_dataset()
        
    def create_dataset(self) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with configurable options
        
        Returns:
            Configured TensorFlow dataset with shuffling, preprocessing, batching
        """
        # Dynamically determine output signature based on data type
        if self._y is not None:
            output_signature = (
                tf.TensorSpec(shape=(None), dtype=tf.float32),  # x
                tf.TensorSpec(shape=(None), dtype=tf.float32)  # y                              
            )
        else:
            output_signature = tf.TensorSpec(
                shape=self.get_data_dimensions(), 
                dtype=tf.float32
            )
        
        # Create dataset using generator
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            self.dataset_generator_function, 
            output_signature=output_signature
        )
        
        # Optional: shuffle the order in which the data is presented to the model 
        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=len(self._x))
            
        # Create batches and prefetch
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._num_parallel_calls)
        
        return dataset
    
    def dataset_generator_function(self):
        """
        Generator function for creating dataset
        
        Yields:
            Individual samples or (sample, label) tuples
        """
        indices = np.arange(len(self._x))
        
        # Optional: shuffle the internal order of the data, so that batches are created from un-ordered data
        if self._shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            
            sample = self._x[idx]         

            if self._y is not None:
                yield sample, self._y[idx]
            else:
                yield sample

    def get_fold_dataset(self, fold_indices):
        """
        Create a TensorFlow dataset using only the specified indices.

        This method is particularly useful when working with large datasets that cannot 
        be fully loaded into memory simultaneously. It allows creating a subset of the 
        original dataset based on provided indices.

        Args:
            fold_indices (List[int]): A list of integer indices to select specific 
                samples from the original dataset. These indices correspond to the 
                original input data's indices.

        Returns:
            tf.data.Dataset: A configured TensorFlow dataset containing only the 
                samples corresponding to the specified indices. The dataset is prepared 
                with batching and optional shuffling.

        Raises:
            IndexError: If any of the provided indices are out of bounds.

        Example:
            >>> loader = DataLoader(x_data, y_data)
            >>> subset_indices = [0, 2, 4, 6]  # Select specific samples
            >>> fold_dataset = loader.get_fold_dataset(subset_indices)
        """
        # Existing implementation remains the same
        if self._y is not None:
            output_signature = (
                tf.TensorSpec(shape=self.get_data_dimensions(), dtype=tf.float32),  # x
                tf.TensorSpec(shape=self.get_label_dimensions(), dtype=tf.float32)  # y                              
            )
        else:
            output_signature = tf.TensorSpec(
                shape=self.get_data_dimensions(), 
                dtype=tf.float32
            )

        dataset = tf.data.Dataset.from_generator(
            lambda: self.fold_generator_function(fold_indices),
            output_signature=output_signature
        )

        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        
        dataset = dataset.prefetch(self._num_parallel_calls)
        
        return dataset

    def fold_generator_function(self, fold_indices):
        """
        Generator function to yield samples for a specific fold of the dataset.

        This method creates an iterator that yields individual samples (and their 
        corresponding labels, if available) based on the provided indices. It supports 
        optional shuffling of the selected fold.

        Args:
            fold_indices (List[int]): A list of indices to select specific samples 
                from the original dataset.

        Yields:
            If labeled dataset:
                Tuple[np.ndarray, np.ndarray]: A tuple of (sample, label)
            If unlabeled dataset:
                np.ndarray: Individual samples

        Notes:
            - If shuffling is enabled, the order of samples within the fold will be randomized.
            - The method assumes that all provided indices are valid for the original dataset.

        Raises:
            IndexError: If any index in fold_indices is out of bounds of the original dataset.

        Example:
            >>> loader = DataLoader(x_data, y_data)
            >>> fold_indices = [1, 3, 5, 7]
            >>> for sample, label in loader.fold_generator_function(fold_indices):
            ...     # Process each sample in the fold
            ...     pass
        """
        if self._shuffle:
            np.random.shuffle(fold_indices)

        for idx in fold_indices:
            sample = self._x[idx]         

            if self._y is not None:
                yield sample, self._y[idx]
            else:
                yield sample
                
    def get_data_dimensions(self) -> Tuple:
        """
        Get dimensions of the first data sample
        
        Returns:
            Shape of the first sample in the dataset
        """
        return self._x[0].shape

    def get_label_dimensions(self) -> Tuple:
        """
        Get dimensions of the first label
        
        Returns:
            Shape of the first label in the dataset
        """
        return self._y[0].shape
    
    def __len__(self) -> int:
        """
        Calculate total number of samples in the dataset.
        
        Returns:
            Total number of samples, regardless of batching.
        """
        return len(self._x)

    def get_num_complete_batches(self) -> int:
        """
        Calculate the number of complete batches.
        
        Returns:
            Number of complete batches in the dataset.
        
        Notes:
            - Batches are created with drop_remainder=True
            - Samples that don't form a complete batch are discarded
        """
        return len(self._x) // self._batch_size
    
    def get_num_effective_samples(self) -> int:
        """
        Calculate the number of samples that will be used in training.
        
        Returns:
            Number of samples that form complete batches.
        
        Notes:
            - Returns the number of samples that will actually be processed
            - Samples that don't form a complete batch are excluded
        """
        return self.get_num_batches() * self._batch_size

    def __iter__(self):
        """
        Makes the class iterable, allowing direct iteration over batches
        
        Returns:
            Iterator for the dataset
        """
        return iter(self._dataset)

    def __next__(self):
        """
        Allows getting the next batch directly
        
        Returns:
            Next batch of data
        """
        return next(iter(self._dataset))
    
    # Getters
    def get_dataset(self):
        return self._dataset
    
    def get_x(self):
        return self._x
    
    def get_y(self):
        if self._y is None:
            raise ValueError("No labels available in this dataset")
        return self._y
    
    def get_batch_size(self):
        return self._batch_size
    
    def get_shuffle(self):
        return self._shuffle
    
    def get_num_parallel_calls(self):
        return self._num_parallel_calls
    

    
    