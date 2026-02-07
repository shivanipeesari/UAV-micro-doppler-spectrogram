"""
Main Execution Module
=====================
This is the main entry point for the UAV vs Bird Classification System.
It orchestrates the complete pipeline from data loading to report generation.

System Architecture:
Data Loading → Preprocessing → Model Training → Evaluation → Prediction → Report Generation

Author: B.Tech Major Project
Date: 2026
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Import custom modules
from dataset.dataset_loader import DatasetLoader
from preprocessing.preprocessing import ImagePreprocessor, DataAugmentation
from spectrogram.spectrogram import SpectrogramGenerator
from model.model import UAVBirdCNN
from training.train import ModelTrainer, TrainingVisualizer
from evaluation.evaluate import ModelEvaluator
from database.predict import ImagePredictor, PredictionAnalyzer
from database.database import PredictionDatabase
from reports.report import ReportGenerator, ReportPrinter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uav_bird_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UAVBirdClassificationSystem:
    """
    Main system class that orchestrates the complete UAV/Bird classification pipeline.
    
    Pipeline Steps:
    1. Load Dataset: Load spectrogram images from DIAT-μSAT dataset
    2. Preprocess: Resize, normalize, and augment images
    3. Train Model: Train CNN model on training set
    4. Evaluate: Evaluate model on test set
    5. Predict: Make predictions on new images
    6. Report: Generate comprehensive reports
    """
    
    def __init__(self, config=None):
        """
        Initialize the system.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self._get_default_config()
        self.dataset_loader = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        self.database = None
        self.report_generator = None
        
        logger.info("Initialized UAVBirdClassificationSystem")
    
    def _get_default_config(self):
        """
        Get default configuration.
        
        Returns:
            dict: Default configuration
        """
        return {
            'dataset_path': './dataset',
            'image_size': (128, 128),
            'batch_size': 32,
            'epochs': 50,
            'test_split': 0.2,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'model_save_path': './model/trained_model.h5',
            'db_path': './database/predictions.db',
            'reports_dir': './reports',
            'augment_data': True,
            'augment_factor': 2,
            'verbose': 1
        }
    
    def load_dataset(self):
        """
        Load dataset from disk.
        
        Returns:
            tuple: (images, labels, file_paths)
        """
        logger.info("=" * 70)
        logger.info("STEP 1: LOADING DATASET")
        logger.info("=" * 70)
        
        dataset_path = self.config['dataset_path']
        image_size = self.config['image_size']
        
        self.dataset_loader = DatasetLoader(dataset_path, image_size=image_size)
        images, labels, file_paths = self.dataset_loader.load_dataset()
        
        # Print dataset info
        info = self.dataset_loader.get_dataset_info()
        print("\nDataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return images, labels, file_paths
    
    def preprocess_data(self, images, augment=True):
        """
        Preprocess images.
        
        Args:
            images (np.ndarray): Input images
            augment (bool): Apply data augmentation
        
        Returns:
            np.ndarray: Preprocessed images
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("=" * 70)
        
        preprocessor = ImagePreprocessor(target_size=self.config['image_size'])
        
        logger.info("Preprocessing images...")
        preprocessed = preprocessor.preprocess_batch(
            images,
            normalize=True,
            denoise=True,
            enhance_contrast=True
        )
        
        # Data augmentation
        if augment and self.config['augment_data']:
            logger.info("Applying data augmentation...")
            augmentor = DataAugmentation()
            # Note: augmentation is typically applied during training via Keras
            # This is optional for initial augmentation
        
        logger.info(f"Preprocessing complete. Output shape: {preprocessed.shape}")
        
        return preprocessed
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the CNN model.
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
        
        Returns:
            dict: Training history
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: TRAINING MODEL")
        logger.info("=" * 70)
        
        # Build model
        logger.info("Building CNN model...")
        cnn = UAVBirdCNN(
            input_shape=(self.config['image_size'][0], self.config['image_size'][1], 1),
            num_classes=2
        )
        self.model = cnn.build_model()
        cnn.compile_model(optimizer='adam', loss='categorical_crossentropy')
        
        # Print model summary
        logger.info("Model Architecture:")
        cnn.get_model_summary()
        
        # Train model
        logger.info("Starting model training...")
        self.trainer = ModelTrainer(self.model, self.config['model_save_path'])
        
        history = self.trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=self.config['verbose']
        )
        
        # Visualize training
        logger.info("Visualizing training results...")
        visualizer = TrainingVisualizer(history)
        visualizer.plot_training_history(output_path='./training_history.png')
        visualizer.print_training_summary()
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.ndarray): Test images
            y_test (np.ndarray): Test labels
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: EVALUATING MODEL")
        logger.info("=" * 70)
        
        self.evaluator = ModelEvaluator(self.model, class_names=['UAV', 'Bird'])
        metrics = self.evaluator.evaluate(X_test, y_test)
        
        # Print summary
        self.evaluator.print_evaluation_summary()
        
        # Generate visualizations
        logger.info("Generating evaluation visualizations...")
        self.evaluator.plot_confusion_matrix(output_path='./confusion_matrix.png')
        self.evaluator.plot_roc_curve(output_path='./roc_curve.png')
        self.evaluator.plot_metrics(output_path='./metrics.png')
        
        return metrics
    
    def predict(self, image_paths):
        """
        Make predictions on new images.
        
        Args:
            image_paths (list): List of image file paths
        
        Returns:
            list: Predictions for each image
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: MAKING PREDICTIONS")
        logger.info("=" * 70)
        
        self.predictor = ImagePredictor(self.model, class_names=['UAV', 'Bird'])
        
        if isinstance(image_paths, str):
            # Single image path
            predictions = [self.predictor.predict_single(image_paths)]
        else:
            # Multiple image paths
            predictions = self.predictor.predict_batch(image_paths)
        
        # Analyze predictions
        analyzer = PredictionAnalyzer(predictions)
        analyzer.print_statistics()
        
        return predictions
    
    def store_predictions(self, predictions):
        """
        Store predictions in database.
        
        Args:
            predictions (list): List of predictions
        """
        logger.info("Storing predictions in database...")
        
        self.database = PredictionDatabase(self.config['db_path'])
        self.database.store_batch_predictions(predictions)
        
        # Print database statistics
        stats = self.database.get_statistics()
        logger.info(f"Database Statistics: {stats}")
    
    def generate_reports(self, history=None, metrics=None, predictions=None):
        """
        Generate comprehensive reports.
        
        Args:
            history (dict): Training history
            metrics (dict): Evaluation metrics
            predictions (list): Prediction results
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: GENERATING REPORTS")
        logger.info("=" * 70)
        
        self.report_generator = ReportGenerator(self.config['reports_dir'])
        
        # Generate training report
        if history:
            logger.info("Generating training report...")
            self.report_generator.generate_training_report(history)
        
        # Generate evaluation report
        if metrics:
            logger.info("Generating evaluation report...")
            self.report_generator.generate_evaluation_report(metrics)
        
        # Generate prediction report
        if predictions:
            logger.info("Generating prediction report...")
            self.report_generator.generate_prediction_report(predictions)
            self.report_generator.generate_statistics_report(predictions)
        
        # Generate summary report
        logger.info("Generating summary report...")
        self.report_generator.generate_summary_report(
            model_info={'Architecture': 'CNN', 'Input Size': str(self.config['image_size'])},
            training_metrics={'Epochs': self.config['epochs'], 'Batch Size': self.config['batch_size']},
            evaluation_metrics=metrics
        )
        
        logger.info(f"Reports saved to {self.config['reports_dir']}")
    
    def run_complete_pipeline(self):
        """
        Run the complete classification pipeline.
        
        Returns:
            dict: Summary of results
        """
        logger.info("=" * 70)
        logger.info("STARTING UAV vs BIRD CLASSIFICATION SYSTEM")
        logger.info("=" * 70)
        logger.info(f"Start time: {datetime.now()}")
        
        try:
            # Step 1: Load dataset
            images, labels, file_paths = self.load_dataset()
            
            # Step 2: Preprocess data
            preprocessed_images = self.preprocess_data(images, augment=True)
            
            # Split data
            from sklearn.model_selection import train_test_split
            from tensorflow.keras.utils import to_categorical
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                preprocessed_images, labels,
                test_size=self.config['test_split'],
                stratify=labels,
                random_state=42
            )
            
            val_split = self.config['validation_split'] / (1 - self.config['test_split'])
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_split,
                stratify=y_temp,
                random_state=42
            )
            
            # Add channel dimension
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
            
            # One-hot encode labels
            y_train = to_categorical(y_train, 2)
            y_val = to_categorical(y_val, 2)
            y_test = to_categorical(y_test, 2)
            
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Step 3: Train model
            history = self.train_model(X_train, y_train, X_val, y_val)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Step 5: Generate reports
            self.generate_reports(history=history, metrics=metrics)
            
            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"End time: {datetime.now()}")
            
            return {
                'status': 'success',
                'model': self.model,
                'history': history,
                'metrics': metrics,
                'dataset_info': self.dataset_loader.get_dataset_info() if self.dataset_loader else None
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_prediction_only(self, image_paths):
        """
        Run prediction on new images without training.
        
        Args:
            image_paths (list or str): Image paths for prediction
        
        Returns:
            list: Predictions
        """
        logger.info("=" * 70)
        logger.info("RUNNING PREDICTION ONLY")
        logger.info("=" * 70)
        
        try:
            # Load trained model
            from tensorflow.keras.models import load_model
            if os.path.exists(self.config['model_save_path']):
                logger.info(f"Loading trained model from {self.config['model_save_path']}")
                self.model = load_model(self.config['model_save_path'])
            else:
                logger.error(f"Model not found at {self.config['model_save_path']}")
                return None
            
            # Make predictions
            predictions = self.predict(image_paths)
            
            # Store predictions
            self.store_predictions(predictions)
            
            # Generate prediction report
            self.report_generator = ReportGenerator(self.config['reports_dir'])
            self.report_generator.generate_prediction_report(predictions)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Prediction failed with error: {str(e)}", exc_info=True)
            return None


def main():
    """
    Main entry point.
    """
    # Create system instance
    system = UAVBirdClassificationSystem()
    
    # Run complete pipeline
    # Uncomment the line below to run the full pipeline with actual dataset
    # result = system.run_complete_pipeline()
    
    # For testing without dataset, display system info
    print("\n" + "=" * 70)
    print("UAV vs BIRD CLASSIFICATION SYSTEM")
    print("=" * 70)
    print("\nSystem initialized successfully!")
    print("\nConfiguration:")
    for key, value in system.config.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 70)
    print("Ready to run pipeline. Use run_complete_pipeline() or run_prediction_only()")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
