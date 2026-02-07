"""
CNN Model Module
================
This module defines the Convolutional Neural Network (CNN) architecture
for binary classification of UAVs and Birds using spectrograms.

Author: B.Tech Major Project
Date: 2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UAVBirdCNN:
    """
    CNN model for binary classification of UAVs vs Birds.
    
    Architecture:
    - Input: Grayscale spectrogram image (128x128)
    - Conv2D layers with ReLU activation
    - MaxPooling layers for dimensionality reduction
    - Flatten layer
    - Dense layers with Dropout for regularization
    - Output: Softmax with 2 classes (UAV, Bird)
    
    This architecture is designed to be:
    - Simple and interpretable for academic purposes
    - Effective for spectrogram classification
    - Not overly complex for easy explanation
    """
    
    def __init__(self, input_shape=(128, 128, 1), num_classes=2):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of output classes (2 for UAV/Bird)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        logger.info(f"Initialized UAVBirdCNN with input shape: {input_shape}, classes: {num_classes}")
    
    def build_model(self):
        """
        Build the CNN model architecture.
        
        Architecture Details:
        - Conv Block 1: 32 filters, 3x3 kernel
        - MaxPool: 2x2
        - Conv Block 2: 64 filters, 3x3 kernel
        - MaxPool: 2x2
        - Conv Block 3: 128 filters, 3x3 kernel
        - MaxPool: 2x2
        - Flatten
        - Dense: 128 units with Dropout
        - Dense: 64 units with Dropout
        - Output Dense: 2 units with Softmax
        
        Returns:
            keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name='conv1_1'
            ),
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name='conv1_2'
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name='pool1'
            ),
            layers.BatchNormalization(name='bn1'),
            
            # Second convolutional block
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name='conv2_1'
            ),
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name='conv2_2'
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name='pool2'
            ),
            layers.BatchNormalization(name='bn2'),
            
            # Third convolutional block
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name='conv3_1'
            ),
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name='conv3_2'
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name='pool3'
            ),
            layers.BatchNormalization(name='bn3'),
            
            # Global Average Pooling for better generalization
            layers.GlobalAveragePooling2D(name='gap'),
            
            # Fully connected layers
            layers.Dense(
                units=256,
                activation='relu',
                name='fc1'
            ),
            layers.Dropout(rate=0.5, name='dropout1'),
            
            layers.Dense(
                units=128,
                activation='relu',
                name='fc2'
            ),
            layers.Dropout(rate=0.5, name='dropout2'),
            
            layers.Dense(
                units=64,
                activation='relu',
                name='fc3'
            ),
            layers.Dropout(rate=0.3, name='dropout3'),
            
            # Output layer
            layers.Dense(
                units=self.num_classes,
                activation='softmax',
                name='output'
            )
        ])
        
        self.model = model
        logger.info("CNN model architecture built successfully")
        
        return model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', 
                      metrics=['accuracy']):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            optimizer (str): Optimizer to use ('adam', 'rmsprop', 'sgd')
            loss (str): Loss function ('categorical_crossentropy' for multi-class)
            metrics (list): Metrics to track during training
        
        Returns:
            keras.Model: Compiled model
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer={optimizer}, loss={loss}")
        
        return self.model
    
    def get_model_summary(self):
        """
        Get and print model summary.
        
        Returns:
            None (prints summary)
        """
        if self.model is None:
            self.build_model()
        
        self.model.summary()
    
    def get_model(self):
        """
        Get the compiled model.
        
        Returns:
            keras.Model: The CNN model
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        return self.model
    
    @staticmethod
    def build_simple_model(input_shape=(128, 128, 1), num_classes=2):
        """
        Build a simpler CNN model for faster training and easier understanding.
        
        This is an alternative lightweight architecture.
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of output classes
        
        Returns:
            keras.Model: Compiled simple CNN model
        """
        model = models.Sequential([
            # Input and first conv block
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Simple CNN model built and compiled")
        
        return model
    
    @staticmethod
    def build_transfer_learning_model(input_shape=(128, 128, 1), num_classes=2):
        """
        Build a transfer learning model using MobileNetV2.
        
        Note: This is a more advanced approach but included for reference.
        The basic CNN model is recommended for academic projects.
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of output classes
        
        Returns:
            keras.Model: Compiled transfer learning model
        """
        # MobileNetV2 expects 3-channel RGB input
        # So we need to adjust if using grayscale
        if input_shape[2] == 1:
            # Convert grayscale to RGB by repeating channels
            input_layer = layers.Input(shape=input_shape)
            x = layers.concatenate([input_layer, input_layer, input_layer], axis=-1)
            
            # Load pre-trained MobileNetV2
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(input_shape[0], input_shape[1], 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            input_layer = layers.Input(shape=input_shape)
            x = input_layer
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Transfer learning model (MobileNetV2) built and compiled")
        
        return model


def main():
    """
    Example usage of UAVBirdCNN model.
    """
    print("=" * 60)
    print("UAV vs Bird Classification CNN Model")
    print("=" * 60)
    
    # Build model
    print("\nBuilding CNN model...")
    cnn = UAVBirdCNN(input_shape=(128, 128, 1), num_classes=2)
    model = cnn.build_model()
    
    # Compile model
    print("\nCompiling model...")
    cnn.compile_model()
    
    # Print model summary
    print("\nModel Summary:")
    cnn.get_model_summary()
    
    # Test with dummy data
    print("\n\nTesting model with dummy data...")
    dummy_input = np.random.randn(10, 128, 128, 1).astype(np.float32)
    dummy_output = model.predict(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print(f"Sample predictions: {dummy_output[0]}")
    print("\nModel built and tested successfully!")


if __name__ == "__main__":
    main()
