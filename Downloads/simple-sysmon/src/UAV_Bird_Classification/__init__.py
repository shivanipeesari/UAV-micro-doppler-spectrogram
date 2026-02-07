"""
UAV Bird Classification Package
================================
A comprehensive Deep Learning system for classifying UAVs and Birds
using radar micro-Doppler spectrogram analysis.

Version: 1.0
Author: B.Tech Major Project
Date: 2026
"""

__version__ = "1.0.0"
__author__ = "B.Tech Major Project"
__date__ = "2026"

# Import main classes for easy access
try:
    from main import UAVBirdClassificationSystem
    from model.model import UAVBirdCNN
    from dataset.dataset_loader import DatasetLoader
    from preprocessing.preprocessing import ImagePreprocessor, DataAugmentation
    from spectrogram.spectrogram import SpectrogramGenerator
    from training.train import ModelTrainer, TrainingVisualizer
    from evaluation.evaluate import ModelEvaluator
    from database.predict import ImagePredictor
    from database.database import PredictionDatabase
    from reports.report import ReportGenerator
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

__all__ = [
    'UAVBirdClassificationSystem',
    'UAVBirdCNN',
    'DatasetLoader',
    'ImagePreprocessor',
    'DataAugmentation',
    'SpectrogramGenerator',
    'ModelTrainer',
    'TrainingVisualizer',
    'ModelEvaluator',
    'ImagePredictor',
    'PredictionDatabase',
    'ReportGenerator'
]
