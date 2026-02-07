"""
Evaluation Module
=================
This module handles evaluation of the trained CNN model.
Includes metrics calculation, confusion matrix, and performance visualization.

Author: B.Tech Major Project
Date: 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates the trained CNN model on test data.
    
    Metrics computed:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
    - ROC-AUC
    """
    
    def __init__(self, model, class_names=['UAV', 'Bird']):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model (keras.Model): Trained CNN model
            class_names (list): Names of the classes
        """
        self.model = model
        self.class_names = class_names
        self.predictions = None
        self.probabilities = None
        self.ground_truth = None
        self.metrics = {}
        
        logger.info(f"Initialized ModelEvaluator with classes: {class_names}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test (np.ndarray): Test images (N, H, W) or (N, H, W, 1)
            y_test (np.ndarray): Test labels (integer encoding or one-hot)
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Ensure proper format
        if len(X_test.shape) == 3:
            X_test = X_test[..., np.newaxis]
        
        # Convert one-hot to integer if necessary
        if len(y_test.shape) == 2:
            y_test_int = np.argmax(y_test, axis=1)
        else:
            y_test_int = y_test
        
        # Get predictions
        self.probabilities = self.model.predict(X_test, verbose=0)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.ground_truth = y_test_int
        
        # Compute metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test_int, self.predictions),
            'precision': precision_score(y_test_int, self.predictions, zero_division=0),
            'recall': recall_score(y_test_int, self.predictions, zero_division=0),
            'f1_score': f1_score(y_test_int, self.predictions, zero_division=0),
            'roc_auc': self._compute_roc_auc(y_test_int)
        }
        
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        logger.info(f"F1-score: {self.metrics['f1_score']:.4f}")
        
        return self.metrics
    
    def _compute_roc_auc(self, y_true):
        """
        Compute ROC-AUC score.
        
        Args:
            y_true (np.ndarray): Ground truth labels
        
        Returns:
            float: ROC-AUC score
        """
        try:
            # For binary classification, use probability of positive class
            y_proba = self.probabilities[:, 1]
            roc_auc = roc_auc_score(y_true, y_proba)
            return roc_auc
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {str(e)}")
            return None
    
    def get_confusion_matrix(self):
        """
        Get confusion matrix.
        
        Returns:
            np.ndarray: Confusion matrix
        """
        if self.predictions is None or self.ground_truth is None:
            logger.error("Predictions not available. Run evaluate() first.")
            return None
        
        cm = confusion_matrix(self.ground_truth, self.predictions)
        return cm
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            str: Classification report
        """
        if self.predictions is None or self.ground_truth is None:
            logger.error("Predictions not available. Run evaluate() first.")
            return None
        
        report = classification_report(
            self.ground_truth,
            self.predictions,
            target_names=self.class_names,
            digits=4
        )
        return report
    
    def plot_confusion_matrix(self, output_path='./confusion_matrix.png', normalize=False):
        """
        Plot confusion matrix.
        
        Args:
            output_path (str): Path to save the plot
            normalize (bool): Normalize confusion matrix
        """
        cm = self.get_confusion_matrix()
        
        if cm is None:
            logger.error("Cannot plot confusion matrix")
            return
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
        plt.close()
    
    def plot_roc_curve(self, output_path='./roc_curve.png'):
        """
        Plot ROC curve.
        
        Args:
            output_path (str): Path to save the plot
        """
        if self.probabilities is None or self.ground_truth is None:
            logger.error("Predictions not available")
            return
        
        try:
            # Compute ROC curve
            y_proba = self.probabilities[:, 1]
            fpr, tpr, _ = roc_curve(self.ground_truth, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {output_path}")
            plt.close()
        
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
    
    def plot_metrics(self, output_path='./metrics.png'):
        """
        Plot evaluation metrics as bar chart.
        
        Args:
            output_path (str): Path to save the plot
        """
        if not self.metrics:
            logger.error("Metrics not available. Run evaluate() first.")
            return
        
        metrics_names = list(self.metrics.keys())
        metrics_values = list(self.metrics.values())
        
        # Filter out None values
        metrics_names = [name for name, val in zip(metrics_names, metrics_values) if val is not None]
        metrics_values = [val for val in metrics_values if val is not None]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.ylabel('Score')
        plt.title('Model Evaluation Metrics')
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {output_path}")
        plt.close()
    
    def print_evaluation_summary(self):
        """
        Print summary of evaluation results.
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        if self.metrics:
            print("\nMetrics:")
            for metric_name, metric_value in self.metrics.items():
                if metric_value is not None:
                    print(f"  {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
        
        print("\nConfusion Matrix:")
        cm = self.get_confusion_matrix()
        if cm is not None:
            print(cm)
        
        print("\nClassification Report:")
        report = self.get_classification_report()
        if report is not None:
            print(report)
        
        print("=" * 60 + "\n")
    
    def get_metrics(self):
        """
        Get computed metrics.
        
        Returns:
            dict: Evaluation metrics
        """
        return self.metrics
    
    def get_predictions(self):
        """
        Get model predictions.
        
        Returns:
            np.ndarray: Predicted class labels
        """
        return self.predictions
    
    def get_probabilities(self):
        """
        Get prediction probabilities.
        
        Returns:
            np.ndarray: Probability predictions
        """
        return self.probabilities


def main():
    """
    Example usage of ModelEvaluator.
    """
    from model.model import UAVBirdCNN
    
    print("Evaluation Module Test")
    print("=" * 60)
    
    # Build and compile model
    print("Building model...")
    cnn = UAVBirdCNN(input_shape=(128, 128, 1), num_classes=2)
    model = cnn.build_model()
    cnn.compile_model()
    
    # Create dummy test data
    print("Creating dummy test data...")
    X_test = np.random.randn(30, 128, 128, 1).astype(np.float32)
    y_test = np.random.randint(0, 2, 30)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model, class_names=['UAV', 'Bird'])
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    # Plot visualizations
    print("\nGenerating visualization plots...")
    evaluator.plot_confusion_matrix(output_path='confusion_matrix.png')
    evaluator.plot_roc_curve(output_path='roc_curve.png')
    evaluator.plot_metrics(output_path='metrics.png')
    
    print("Evaluation test completed!")


if __name__ == "__main__":
    main()
