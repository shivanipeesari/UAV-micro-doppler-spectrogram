"""
Preprocessing Module
====================
This module handles preprocessing of micro-Doppler spectrogram images
including resizing, normalization, noise handling, and data augmentation.

Author: B.Tech Major Project
Date: 2026
"""

import numpy as np
import cv2
from scipy import ndimage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles preprocessing of spectrogram images.
    
    Operations:
    - Resizing to standard dimensions
    - Normalization
    - Noise reduction (Gaussian blur, morphological operations)
    - Contrast enhancement
    - Data augmentation (rotation, flipping)
    """
    
    def __init__(self, target_size=(128, 128)):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            target_size (tuple or int): Target image size (height, width) or int for square
        """
        # Handle both int and tuple formats
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        logger.info(f"Initialized ImagePreprocessor with target size: {self.target_size}")
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target dimensions.
        
        Args:
            image (np.ndarray): Input image
            target_size (tuple): Target size (height, width)
        
        Returns:
            np.ndarray: Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        resized = cv2.resize(image, (target_size[1], target_size[0]))
        return resized
    
    def normalize_image(self, image, method='minmax'):
        """
        Normalize image to [0, 1] range.
        
        Args:
            image (np.ndarray): Input image
            method (str): Normalization method ('minmax' or 'zscore')
        
        Returns:
            np.ndarray: Normalized image
        """
        if method == 'minmax':
            # Min-Max normalization
            image_min = np.min(image)
            image_max = np.max(image)
            if image_max - image_min != 0:
                normalized = (image - image_min) / (image_max - image_min)
            else:
                normalized = image
        
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            if std != 0:
                normalized = (image - mean) / std
            else:
                normalized = image
        
        return normalized.astype(np.float32)
    
    def reduce_noise(self, image, method='gaussian'):
        """
        Reduce noise from image using different filtering techniques.
        
        Args:
            image (np.ndarray): Input image (values in [0, 1])
            method (str): Noise reduction method ('gaussian', 'bilateral', 'morphological')
        
        Returns:
            np.ndarray: Denoised image
        """
        # Convert to [0, 255] for OpenCV
        image_cv = (image * 255).astype(np.uint8)
        
        if method == 'gaussian':
            # Gaussian blur
            denoised = cv2.GaussianBlur(image_cv, (5, 5), 0)
        
        elif method == 'bilateral':
            # Bilateral filter (preserves edges)
            denoised = cv2.bilateralFilter(image_cv, 9, 75, 75)
        
        elif method == 'morphological':
            # Morphological operations (opening)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            denoised = cv2.morphologyEx(image_cv, cv2.MORPH_OPEN, kernel)
        
        else:
            denoised = image_cv
        
        # Convert back to [0, 1]
        denoised = denoised.astype(np.float32) / 255.0
        return denoised
    
    def enhance_contrast(self, image, method='clahe'):
        """
        Enhance contrast of image.
        
        Args:
            image (np.ndarray): Input image (values in [0, 1])
            method (str): Contrast enhancement method ('clahe', 'histogram')
        
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to [0, 255] for OpenCV
        image_cv = (image * 255).astype(np.uint8)
        
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_cv)
        
        elif method == 'histogram':
            # Standard histogram equalization
            enhanced = cv2.equalizeHist(image_cv)
        
        else:
            enhanced = image_cv
        
        # Convert back to [0, 1]
        enhanced = enhanced.astype(np.float32) / 255.0
        return enhanced
    
    def rotate_image(self, image, angle):
        """
        Rotate image by given angle.
        
        Args:
            image (np.ndarray): Input image
            angle (float): Rotation angle in degrees
        
        Returns:
            np.ndarray: Rotated image
        """
        # Convert to [0, 255] for OpenCV
        image_cv = (image * 255).astype(np.uint8)
        
        h, w = image_cv.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_cv, matrix, (w, h))
        
        # Convert back to [0, 1]
        rotated = rotated.astype(np.float32) / 255.0
        return rotated
    
    def flip_image(self, image, direction='horizontal'):
        """
        Flip image horizontally or vertically.
        
        Args:
            image (np.ndarray): Input image
            direction (str): 'horizontal' or 'vertical'
        
        Returns:
            np.ndarray: Flipped image
        """
        if direction == 'horizontal':
            flipped = np.fliplr(image)
        elif direction == 'vertical':
            flipped = np.flipud(image)
        else:
            flipped = image
        
        return flipped
    
    def preprocess(self, image, normalize=True, denoise=True, enhance_contrast=False):
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image (np.ndarray): Input image
            normalize (bool): Apply normalization
            denoise (bool): Apply noise reduction
            enhance_contrast (bool): Apply contrast enhancement
        
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize
        processed = self.resize_image(image)
        
        # Normalize
        if normalize:
            processed = self.normalize_image(processed)
        
        # Denoise
        if denoise:
            processed = self.reduce_noise(processed, method='gaussian')
        
        # Enhance contrast
        if enhance_contrast:
            processed = self.enhance_contrast(processed, method='clahe')
        
        return processed
    
    def preprocess_batch(self, images, normalize=True, denoise=True, enhance_contrast=False):
        """
        Apply preprocessing to a batch of images.
        
        Args:
            images (np.ndarray): Batch of images (N, H, W) or (N, H, W, C)
            normalize (bool): Apply normalization
            denoise (bool): Apply noise reduction
            enhance_contrast (bool): Apply contrast enhancement
        
        Returns:
            np.ndarray: Preprocessed batch
        """
        processed_images = []
        
        for i, image in enumerate(images):
            if i % 100 == 0:
                logger.info(f"Processing image {i}/{len(images)}")
            
            processed = self.preprocess(
                image,
                normalize=normalize,
                denoise=denoise,
                enhance_contrast=enhance_contrast
            )
            processed_images.append(processed)
        
        return np.array(processed_images)


class DataAugmentation:
    """
    Handles data augmentation for training dataset.
    
    Operations:
    - Random rotation
    - Random flipping
    - Random noise injection
    - Random intensity adjustment
    """
    
    def __init__(self):
        """Initialize DataAugmentation."""
        logger.info("Initialized DataAugmentation")
    
    def add_gaussian_noise(self, image, noise_std=0.01):
        """
        Add Gaussian noise to image.
        
        Args:
            image (np.ndarray): Input image (values in [0, 1])
            noise_std (float): Standard deviation of noise
        
        Returns:
            np.ndarray: Image with added noise
        """
        noise = np.random.normal(0, noise_std, image.shape)
        noisy_image = image + noise
        # Clip to [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image
    
    def adjust_intensity(self, image, factor=1.1):
        """
        Adjust image intensity (brightness).
        
        Args:
            image (np.ndarray): Input image (values in [0, 1])
            factor (float): Intensity factor (>1 brightens, <1 darkens)
        
        Returns:
            np.ndarray: Intensity-adjusted image
        """
        adjusted = image * factor
        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted
    
    def augment_image(self, image, rotation=True, flip=True, noise=True, intensity=True):
        """
        Apply random augmentation to image.
        
        Args:
            image (np.ndarray): Input image
            rotation (bool): Apply random rotation
            flip (bool): Apply random flip
            noise (bool): Add random noise
            intensity (bool): Adjust intensity
        
        Returns:
            list: Original image and augmented copies
        """
        augmented = [image]
        preprocessor = ImagePreprocessor()
        
        # Random rotation
        if rotation:
            angle = np.random.uniform(-15, 15)
            rotated = preprocessor.rotate_image(image, angle)
            augmented.append(rotated)
        
        # Random flip
        if flip:
            flipped = preprocessor.flip_image(image, direction='horizontal')
            augmented.append(flipped)
        
        # Add noise
        if noise:
            noisy = self.add_gaussian_noise(image, noise_std=0.01)
            augmented.append(noisy)
        
        # Adjust intensity
        if intensity:
            bright = self.adjust_intensity(image, factor=1.2)
            dark = self.adjust_intensity(image, factor=0.8)
            augmented.extend([bright, dark])
        
        return augmented
    
    def augment_dataset(self, images, labels, augment_factor=2):
        """
        Augment entire dataset by creating multiple augmented copies.
        
        Args:
            images (np.ndarray): Input images
            labels (np.ndarray): Corresponding labels
            augment_factor (int): Number of augmented copies per image
        
        Returns:
            tuple: (augmented_images, augmented_labels)
        """
        augmented_images = [images]
        augmented_labels = [labels]
        
        for i in range(augment_factor):
            logger.info(f"Augmentation iteration {i+1}/{augment_factor}")
            
            aug_images = []
            for j, image in enumerate(images):
                # Apply augmentation with randomness
                augmented = self.augment_image(
                    image,
                    rotation=True,
                    flip=np.random.choice([True, False]),
                    noise=np.random.choice([True, False]),
                    intensity=True
                )
                # Take one random augmented version
                aug_images.append(augmented[np.random.randint(0, len(augmented))])
            
            augmented_images.append(np.array(aug_images))
            augmented_labels.append(labels.copy())
        
        # Concatenate all augmented data
        final_images = np.concatenate(augmented_images, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        logger.info(f"Dataset augmented from {len(images)} to {len(final_images)} images")
        
        return final_images, final_labels


def main():
    """
    Example usage of preprocessing and augmentation.
    """
    # Create synthetic test image
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    
    # Preprocess image
    processed = preprocessor.preprocess(
        test_image,
        normalize=True,
        denoise=True,
        enhance_contrast=True
    )
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Processed image shape: {processed.shape}")
    
    # Initialize augmentation
    augmentor = DataAugmentation()
    
    # Create synthetic batch
    batch = np.random.rand(10, 256, 256).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Augment dataset
    aug_images, aug_labels = augmentor.augment_dataset(batch, labels, augment_factor=2)
    
    print(f"\nOriginal batch size: {batch.shape}")
    print(f"Augmented batch size: {aug_images.shape}")


if __name__ == "__main__":
    main()
