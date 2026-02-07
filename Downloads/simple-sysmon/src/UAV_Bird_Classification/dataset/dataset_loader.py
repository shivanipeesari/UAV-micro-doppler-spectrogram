"""
Dataset Loader Module
=====================
This module handles loading and organizing micro-Doppler spectrogram images
from the DIAT-μSAT dataset for UAV and Bird classification.

Author: B.Tech Major Project
Date: 2026
"""

import os
import numpy as np
import cv2
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads micro-Doppler spectrogram images from the DIAT-μSAT dataset.
    
    This class handles:
    - Loading images from organized folders (UAV, Bird)
    - Converting images to numpy arrays
    - Splitting data into training and testing sets
    - Organizing labels and file paths
    """
    
    def __init__(self, dataset_path, image_size=128):
        """
        Initialize the DatasetLoader.
        
        Args:
            dataset_path (str): Path to the dataset root directory
            image_size (int or tuple): Target image size. If int, will create square image
        """
        self.dataset_path = Path(dataset_path)
        # Handle image_size - convert to tuple if needed
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.classes = ['UAV', 'Bird']  # Binary classification
        self.images = []
        self.labels = []
        self.file_paths = []
        
        logger.info(f"Initialized DatasetLoader with path: {self.dataset_path}")
    
    def load_dataset(self):
        """
        Load all spectrogram images from the dataset.
        
        Expected directory structure:
        dataset/
        ├── UAV/
        │   ├── image1.png
        │   ├── image2.jpg
        │   └── ...
        └── Bird/
            ├── image1.png
            ├── image2.jpg
            └── ...
        
        Returns:
            tuple: (images_array, labels_array, file_paths)
        """
        logger.info("Loading dataset...")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                logger.warning(f"Class directory not found: {class_path}")
                continue
            
            logger.info(f"Loading {class_name} images from {class_path}...")
            
            # Get all image files (png, jpg, jpeg, bmp)
            image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG')
            image_files = []
            for ext in image_extensions:
                image_files.extend(class_path.glob(ext))
            
            logger.info(f"Found {len(image_files)} {class_name} images")
            
            for image_path in image_files:
                try:
                    # Read image in grayscale
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    
                    if image is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    # Resize image to target size - cv2.resize expects (width, height)
                    image_resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))
                    
                    # Normalize image to [0, 1] range
                    image_normalized = image_resized.astype(np.float32) / 255.0
                    
                    self.images.append(image_normalized)
                    self.labels.append(class_idx)
                    self.file_paths.append(str(image_path))
                    
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {str(e)}")
                    continue
        
        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Total images: {len(self.images)}")
        logger.info(f"UAV images: {np.sum(self.labels == 0)}")
        logger.info(f"Bird images: {np.sum(self.labels == 1)}")
        logger.info(f"Image shape: {self.images[0].shape if len(self.images) > 0 else 'N/A'}")
        
        return self.images, self.labels, self.file_paths
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets.
        
        Args:
            test_size (float): Fraction of data to use for testing (default: 0.2)
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if len(self.images) == 0:
            logger.error("Dataset is empty. Load dataset first using load_dataset().")
            return None
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.images,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_class_names(self):
        """
        Get class names.
        
        Returns:
            list: ['UAV', 'Bird']
        """
        return self.classes
    
    def get_dataset_info(self):
        """
        Get dataset information.
        
        Returns:
            dict: Information about the loaded dataset
        """
        return {
            'total_samples': len(self.images),
            'uav_samples': int(np.sum(self.labels == 0)),
            'bird_samples': int(np.sum(self.labels == 1)),
            'image_shape': self.images[0].shape if len(self.images) > 0 else None,
            'classes': self.classes,
            'target_size': self.image_size
        }


def main():
    """
    Example usage of DatasetLoader.
    """
    # Define dataset path (modify based on your dataset location)
    dataset_path = "./dataset"
    
    # Initialize loader
    loader = DatasetLoader(dataset_path, image_size=(128, 128))
    
    # Load dataset
    images, labels, file_paths = loader.load_dataset()
    
    # Get dataset information
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get train-test split
    X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")


if __name__ == "__main__":
    main()
