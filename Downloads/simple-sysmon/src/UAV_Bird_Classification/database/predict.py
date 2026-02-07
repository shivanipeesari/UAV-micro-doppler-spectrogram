"""
Prediction Module
=================
This module handles prediction on new spectrogram images
and provides confidence scores for UAV vs Bird classification.

Author: B.Tech Major Project
Date: 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePredictor:
    """
    Performs classification predictions on new spectrogram images.
    
    Features:
    - Single image prediction
    - Batch prediction
    - Confidence score computation
    - Class probability output
    """
    
    def __init__(self, model, class_names=['UAV', 'Bird'], 
                 input_size=(128, 128), confidence_threshold=0.5):
        """
        Initialize the ImagePredictor.
        
        Args:
            model (keras.Model): Trained CNN model
            class_names (list): List of class names
            input_size (tuple): Expected input image size
            confidence_threshold (float): Minimum confidence for prediction
        """
        self.model = model
        self.class_names = class_names
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Initialized ImagePredictor with classes: {class_names}")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for prediction.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            np.ndarray: Preprocessed image (H, W, 1)
        """
        try:
            # Read image in grayscale
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize to expected input size
            image_resized = cv2.resize(image, self.input_size)
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Add channel dimension
            image_with_channel = image_normalized[..., np.newaxis]
            
            logger.info(f"Image preprocessed: {image_with_channel.shape}")
            
            return image_with_channel
        
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def predict_single(self, image_path):
        """
        Predict class for a single image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Prediction result with class, confidence, and probabilities
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        if image is None:
            return {
                'class': 'Error',
                'confidence': 0.0,
                'probabilities': {},
                'image_path': str(image_path)
            }
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Get prediction
        probabilities = self.model.predict(image_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        # Create result
        result = {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'image_path': str(image_path),
            'meets_threshold': confidence >= self.confidence_threshold
        }
        
        logger.info(f"Prediction for {image_path}: {result['class']} (confidence: {confidence:.4f})")
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict class for multiple images.
        
        Args:
            image_paths (list): List of image file paths
        
        Returns:
            list: List of prediction results
        """
        logger.info(f"Predicting on {len(image_paths)} images...")
        
        results = []
        successful_predictions = 0
        
        for i, image_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images")
            
            result = self.predict_single(image_path)
            results.append(result)
            
            if result['class'] != 'Error':
                successful_predictions += 1
        
        logger.info(f"Batch prediction completed. Successful: {successful_predictions}/{len(image_paths)}")
        
        return results
    
    def predict_from_array(self, image_array):
        """
        Predict class from numpy array.
        
        Args:
            image_array (np.ndarray): Image array (H, W) or (H, W, 1)
        
        Returns:
            dict: Prediction result
        """
        # Ensure proper shape
        if len(image_array.shape) == 2:
            image_array = image_array[..., np.newaxis]
        
        # Resize if necessary
        if image_array.shape[:2] != self.input_size:
            image_resized = cv2.resize(image_array, self.input_size)
        else:
            image_resized = image_array
        
        # Normalize if not already
        if image_resized.max() > 1.0:
            image_resized = image_resized.astype(np.float32) / 255.0
        else:
            image_resized = image_resized.astype(np.float32)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_resized, axis=0)
        
        # Get prediction
        probabilities = self.model.predict(image_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        result = {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'meets_threshold': confidence >= self.confidence_threshold
        }
        
        return result
    
    def predict_from_directory(self, directory_path):
        """
        Predict class for all images in a directory.
        
        Args:
            directory_path (str): Path to directory containing images
        
        Returns:
            dict: Dictionary with predictions organized by class
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return {}
        
        # Get all image files
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(ext))
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        # Predict on all images
        results = self.predict_batch(image_files)
        
        # Organize by predicted class
        predictions_by_class = {class_name: [] for class_name in self.class_names}
        
        for result in results:
            if result['class'] in predictions_by_class:
                predictions_by_class[result['class']].append(result)
        
        logger.info(f"Predictions summary:")
        for class_name, predictions in predictions_by_class.items():
            logger.info(f"  {class_name}: {len(predictions)} images")
        
        return predictions_by_class
    
    def set_confidence_threshold(self, threshold):
        """
        Set confidence threshold for predictions.
        
        Args:
            threshold (float): Confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold:.4f}")


class PredictionAnalyzer:
    """
    Analyzes prediction results and provides statistics.
    """
    
    def __init__(self, predictions):
        """
        Initialize analyzer with predictions.
        
        Args:
            predictions (list): List of prediction results
        """
        self.predictions = predictions
    
    def get_statistics(self):
        """
        Get statistics from predictions.
        
        Returns:
            dict: Statistical summary
        """
        if not self.predictions:
            return {}
        
        successful = [p for p in self.predictions if p['class'] != 'Error']
        
        if not successful:
            return {'error': 'No successful predictions'}
        
        class_counts = {}
        confidence_by_class = {}
        
        for prediction in successful:
            class_name = prediction['class']
            confidence = prediction['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                confidence_by_class[class_name] = []
            
            class_counts[class_name] += 1
            confidence_by_class[class_name].append(confidence)
        
        # Compute averages
        avg_confidence = {}
        for class_name, confidences in confidence_by_class.items():
            avg_confidence[class_name] = np.mean(confidences)
        
        stats = {
            'total_predictions': len(self.predictions),
            'successful_predictions': len(successful),
            'class_counts': class_counts,
            'average_confidence': avg_confidence,
            'overall_confidence': np.mean([p['confidence'] for p in successful])
        }
        
        return stats
    
    def print_statistics(self):
        """
        Print prediction statistics.
        """
        stats = self.get_statistics()
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return
        
        print("\n" + "=" * 60)
        print("PREDICTION STATISTICS")
        print("=" * 60)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Successful predictions: {stats['successful_predictions']}")
        print(f"\nClass Distribution:")
        for class_name, count in stats['class_counts'].items():
            percentage = (count / stats['successful_predictions']) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"\nAverage Confidence by Class:")
        for class_name, avg_conf in stats['average_confidence'].items():
            print(f"  {class_name}: {avg_conf:.4f}")
        print(f"\nOverall Average Confidence: {stats['overall_confidence']:.4f}")
        print("=" * 60 + "\n")


def main():
    """
    Example usage of ImagePredictor.
    """
    from model.model import UAVBirdCNN
    
    print("Prediction Module Test")
    print("=" * 60)
    
    # Build and compile model
    print("Building model...")
    cnn = UAVBirdCNN(input_shape=(128, 128, 1), num_classes=2)
    model = cnn.build_model()
    cnn.compile_model()
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = ImagePredictor(model, class_names=['UAV', 'Bird'])
    
    # Create and test with dummy data
    print("Creating dummy image array...")
    dummy_image = np.random.rand(128, 128).astype(np.float32)
    
    result = predictor.predict_from_array(dummy_image)
    
    print("\nPrediction Result:")
    print(f"  Class: {result['class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Probabilities: {result['probabilities']}")
    
    print("\nPrediction test completed!")


if __name__ == "__main__":
    main()
