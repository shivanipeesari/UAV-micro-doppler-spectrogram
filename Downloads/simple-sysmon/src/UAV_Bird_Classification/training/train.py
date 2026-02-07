"""
Training Module
===============
This module handles the training of the CNN model on the UAV/Bird dataset.

Author: B.Tech Major Project
Date: 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training of the CNN model.
    
    Features:
    - Training with validation
    - Early stopping to prevent overfitting
    - Learning rate reduction on plateau
    - Model checkpointing
    - Training history tracking
    """
    
    def __init__(self, model, model_save_path='./model/trained_model.h5'):
        """
        Initialize the ModelTrainer.
        
        Args:
            model (keras.Model): The CNN model to train
            model_save_path (str): Path to save the trained model
        """
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
        self.best_epoch = None
        self.best_accuracy = 0.0
        
        logger.info(f"Initialized ModelTrainer with save path: {model_save_path}")
    
    def get_callbacks(self, checkpoint_dir='./model'):
        """
        Get callbacks for training.
        
        Callbacks used:
        - EarlyStopping: Stop training if validation loss doesn't improve
        - ReduceLROnPlateau: Reduce learning rate if metric plateaus
        - ModelCheckpoint: Save best model
        - TensorBoard: Visualization (optional)
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
        
        Returns:
            list: List of keras callbacks
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        logger.info("Callbacks configured")
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the CNN model.
        
        Args:
            X_train (np.ndarray): Training images (N, H, W, 1)
            y_train (np.ndarray): Training labels (one-hot encoded)
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels (one-hot encoded)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level (0, 1, or 2)
        
        Returns:
            dict: Training history
        """
        logger.info("=" * 60)
        logger.info("Starting model training")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        
        # Ensure labels are one-hot encoded
        if len(y_train.shape) == 1:
            logger.info("One-hot encoding training labels...")
            y_train = keras.utils.to_categorical(y_train, num_classes=2)
            y_val = keras.utils.to_categorical(y_val, num_classes=2)
        
        # Ensure images have channel dimension
        if len(X_train.shape) == 3:
            logger.info("Adding channel dimension to images...")
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train the model
        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            logger.info("Training completed successfully!")
            logger.info(f"Final training accuracy: {self.history.history['accuracy'][-1]:.4f}")
            logger.info(f"Final validation accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        return self.history.history
    
    def train_with_augmentation(self, X_train, y_train, X_val, y_val,
                                augmentation_pipeline=None,
                                epochs=50, batch_size=32, verbose=1):
        """
        Train the model with data augmentation.
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
            augmentation_pipeline: Keras preprocessing pipeline
            epochs (int): Number of epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
        
        Returns:
            dict: Training history
        """
        logger.info("Training with data augmentation enabled")
        
        # Ensure labels are one-hot encoded
        if len(y_train.shape) == 1:
            y_train = keras.utils.to_categorical(y_train, num_classes=2)
            y_val = keras.utils.to_categorical(y_val, num_classes=2)
        
        # Ensure images have channel dimension
        if len(X_train.shape) == 3:
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]
        
        # If no pipeline provided, create a default one
        if augmentation_pipeline is None:
            augmentation_pipeline = keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.2),
            ])
        
        # Create augmented dataset
        augmented_images = augmentation_pipeline(X_train, training=True)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train with augmentation
        self.history = self.model.fit(
            augmented_images, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Training with augmentation completed!")
        return self.history.history
    
    def get_training_history(self):
        """
        Get training history.
        
        Returns:
            dict: Training history containing accuracy, loss, etc.
        """
        if self.history is None:
            logger.warning("No training history available. Train model first.")
            return None
        
        return self.history.history
    
    def save_model(self, filepath=None):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if filepath is None:
            filepath = self.model_save_path
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to load the model from
        """
        if filepath is None:
            filepath = self.model_save_path
        
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def get_model_info(self):
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information
        """
        info = {
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) 
                                    for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) 
                                        for w in self.model.non_trainable_weights])
        }
        return info


class TrainingVisualizer:
    """
    Handles visualization of training history.
    """
    
    def __init__(self, history):
        """
        Initialize visualizer with training history.
        
        Args:
            history (dict): Training history from model.fit()
        """
        self.history = history
    
    def plot_training_history(self, output_path='./training_history.png'):
        """
        Plot training accuracy and loss.
        
        Args:
            output_path (str): Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(self.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {output_path}")
        plt.close()
    
    def print_training_summary(self):
        """
        Print summary of training results.
        """
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total epochs trained: {len(self.history['accuracy'])}")
        print(f"Final training accuracy: {self.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {self.history['val_accuracy'][-1]:.4f}")
        print(f"Best validation accuracy: {max(self.history['val_accuracy']):.4f}")
        print(f"Best epoch: {np.argmax(self.history['val_accuracy']) + 1}")
        print(f"Final training loss: {self.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {self.history['val_loss'][-1]:.4f}")
        print("=" * 60 + "\n")


def main():
    """
    Example usage of ModelTrainer.
    """
    from model.model import UAVBirdCNN
    
    print("Training Module Test")
    print("=" * 60)
    
    # Build and compile model
    print("Building model...")
    cnn = UAVBirdCNN(input_shape=(128, 128, 1), num_classes=2)
    model = cnn.build_model()
    cnn.compile_model()
    
    # Create dummy training data
    print("Creating dummy training data...")
    X_train = np.random.randn(100, 128, 128, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 128, 128, 1).astype(np.float32)
    y_val = np.random.randint(0, 2, 20)
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_val = keras.utils.to_categorical(y_val, num_classes=2)
    
    # Train model
    print("\nTraining model...")
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    
    # Visualize training
    print("\nVisualizing training history...")
    visualizer = TrainingVisualizer(history)
    visualizer.print_training_summary()
    visualizer.plot_training_history()
    
    print("Training test completed!")


if __name__ == "__main__":
    main()
