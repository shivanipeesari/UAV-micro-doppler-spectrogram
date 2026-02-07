"""
Advanced Visualization Module
==============================
Generates professional plots and visualizations for the UAV vs Bird classification system.

Includes:
- Training history plots (accuracy, loss, learning curves)
- Confusion matrix heatmap
- ROC curve with AUC
- Precision-Recall curve
- Feature importance analysis
- Prediction confidence distribution
- Model complexity analysis

Author: B.Tech Major Project
Date: 2026
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, f1_score,
    roc_auc_score
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedVisualizer:
    """
    Create professional visualizations for model results.
    """
    
    def __init__(self, output_dir='./reports/'):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = output_dir
        logger.info(f"Initialized AdvancedVisualizer with output dir: {output_dir}")
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training and validation accuracy/loss.
        
        Args:
            history: Keras training history object
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}training_history.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved training history plot to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['UAV', 'Bird'], 
                            save_path=None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names (list): Names of classes
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}confusion_matrix.png"
        
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 14})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - UAV vs Bird Classification', 
                    fontsize=14, fontweight='bold')
        
        # Add text annotations for TP, FP, FN, TN
        ax.text(0.5, -0.2, 
               f'TP={cm[0,0]} FP={cm[1,0]} | FN={cm[0,1]} TN={cm[1,1]}',
               ha='center', fontsize=10, transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve with AUC score.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Predicted probabilities
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}roc_curve.png"
        
        # Convert one-hot to binary
        if len(y_true.shape) > 1:
            y_true_binary = y_true[:, 1]  # Get probability for UAV class
        else:
            y_true_binary = y_true
        
        if len(y_pred_proba.shape) > 1:
            y_pred_binary = y_pred_proba[:, 1]
        else:
            y_pred_binary = y_pred_proba
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - Model Performance', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved ROC curve to {save_path}")
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}precision_recall_curve.png"
        
        # Convert one-hot to binary
        if len(y_true.shape) > 1:
            y_true_binary = y_true[:, 1]
        else:
            y_true_binary = y_true
        
        if len(y_pred_proba.shape) > 1:
            y_pred_binary = y_pred_proba[:, 1]
        else:
            y_pred_binary = y_pred_proba
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_binary)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='blue', lw=2, 
               label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Precision-Recall curve to {save_path}")
        plt.close()
        
        return pr_auc
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of different metrics.
        
        Args:
            metrics_dict: Dictionary with metric names and values
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}metrics_comparison.png"
        
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved metrics comparison to {save_path}")
        plt.close()
    
    def plot_prediction_confidence(self, y_pred_proba, y_true, save_path=None):
        """
        Plot distribution of prediction confidence.
        
        Args:
            y_pred_proba: Predicted probabilities
            y_true: True labels
            save_path (str): Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}prediction_confidence.png"
        
        # Convert one-hot to indices
        if len(y_true.shape) > 1:
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            y_true_indices = y_true
        
        if len(y_pred_proba.shape) > 1:
            y_pred_indices = np.argmax(y_pred_proba, axis=1)
            y_confidence = np.max(y_pred_proba, axis=1)
        else:
            y_pred_indices = y_pred_proba
            y_confidence = np.abs(y_pred_proba)
        
        # Separate correct and incorrect predictions
        correct = y_true_indices == y_pred_indices
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(y_confidence[correct], bins=20, alpha=0.7, label='Correct Predictions', color='green')
        ax.hist(y_confidence[~correct], bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved confidence distribution to {save_path}")
        plt.close()
    
    def plot_learning_curves(self, train_losses, val_losses, train_accs, val_accs, 
                           save_path=None):
        """
        Plot detailed learning curves showing overfitting/underfitting.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            train_accs: Training accuracies per epoch
            val_accs: Validation accuracies per epoch
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}learning_curves.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = np.arange(1, len(train_losses) + 1)
        
        # Training vs Validation Loss
        axes[0, 0].plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training vs Validation Accuracy
        axes[0, 1].plot(epochs, train_accs, 'o-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_accs, 's-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Loss gap (overfitting indicator)
        loss_gap = np.array(val_losses) - np.array(train_losses)
        axes[1, 0].fill_between(epochs, 0, loss_gap, alpha=0.3, color='red')
        axes[1, 0].plot(epochs, loss_gap, 'ro-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Loss Gap (Val - Train)', fontweight='bold')
        axes[1, 0].set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        # Accuracy gap
        acc_gap = np.array(train_accs) - np.array(val_accs)
        axes[1, 1].fill_between(epochs, 0, acc_gap, alpha=0.3, color='orange')
        axes[1, 1].plot(epochs, acc_gap, 'o-', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy Gap (Train - Val)', fontweight='bold')
        axes[1, 1].set_title('Generalization Gap', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        plt.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved learning curves to {save_path}")
        plt.close()
    
    def plot_sample_predictions(self, images, y_true, y_pred, y_pred_proba, 
                               class_names=['UAV', 'Bird'], save_path=None):
        """
        Plot sample images with predictions.
        
        Args:
            images: Sample images (max 9)
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: Class names
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = f"{self.output_dir}sample_predictions.png"
        
        num_samples = min(9, len(images))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_samples):
            img = images[i]
            if img.shape[-1] == 1:
                img = img.squeeze()
            
            axes[i].imshow(img, cmap='gray')
            
            # Get true and predicted class
            if len(y_true.shape) > 1:
                true_label = class_names[np.argmax(y_true[i])]
            else:
                true_label = class_names[y_true[i]]
            
            if len(y_pred.shape) > 1:
                pred_label = class_names[np.argmax(y_pred[i])]
                conf = np.max(y_pred_proba[i])
            else:
                pred_label = class_names[y_pred[i]]
                conf = y_pred_proba[i]
            
            # Color code: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label} ({conf:.2f})',
                fontsize=10, fontweight='bold', color=color
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, 9):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved sample predictions to {save_path}")
        plt.close()


class ModelAnalyzer:
    """
    Analyze model architecture and complexity.
    """
    
    @staticmethod
    def print_model_summary(model):
        """
        Print detailed model summary with parameter counts.
        
        Args:
            model: Keras model
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL ARCHITECTURE SUMMARY")
        logger.info("="*80)
        
        model.summary()
        
        total_params = model.count_params()
        trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
        non_trainable = total_params - trainable_params
        
        logger.info("\n" + "="*80)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Non-trainable Parameters: {non_trainable:,}")
        logger.info("="*80 + "\n")
    
    @staticmethod
    def estimate_model_size(model, input_dtype='float32'):
        """
        Estimate model file size.
        
        Args:
            model: Keras model
            input_dtype: Data type of inputs
        
        Returns:
            float: Size in MB
        """
        # Calculate parameters
        total_params = model.count_params()
        
        # Assume float32 (4 bytes per parameter)
        bytes_per_param = 4
        model_size_bytes = total_params * bytes_per_param
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        return model_size_mb


if __name__ == "__main__":
    logger.info("Advanced Visualization Module loaded successfully")
