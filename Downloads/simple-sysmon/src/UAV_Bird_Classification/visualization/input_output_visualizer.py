"""
Input-Output Visualization Module
==================================
Displays input spectrograms and model predictions side-by-side for 
comprehensive analysis and viva demonstration.

Features:
- Single prediction with input→output display
- Batch analysis with spectrogram grid
- Spectrogram comparison (UAV vs Bird)
- Prediction confidence visualization
- Model interpretation aids

Author: B.Tech Major Project
Date: 2026
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputOutputVisualizer:
    """
    Professional visualization of inputs and model outputs.
    """
    
    def __init__(self, output_dir='./reports/'):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized InputOutputVisualizer: {self.output_dir}")
    
    def visualize_single_prediction(self, image_path, prediction_class, confidence, 
                                   save_path=None, title=None):
        """
        Display input spectrogram and prediction result side-by-side.
        
        Args:
            image_path (str): Path to input spectrogram image
            prediction_class (str): Predicted class ('UAV' or 'Bird')
            confidence (float): Prediction confidence (0-1)
            save_path (str): Path to save visualization
            title (str): Custom title
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Create figure with custom layout
            fig = plt.figure(figsize=(14, 6))
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Input spectrogram (left side, tall)
            ax_input = fig.add_subplot(gs[:, 0])
            ax_input.imshow(img_array)
            ax_input.set_title('Input Spectrogram\n(Radar Micro-Doppler Signature)', 
                              fontsize=12, fontweight='bold', pad=10)
            ax_input.axis('off')
            
            # Prediction result (top right)
            ax_pred = fig.add_subplot(gs[0, 1])
            ax_pred.axis('off')
            
            # Color based on prediction
            color = '#2ecc71' if prediction_class == 'UAV' else '#e74c3c'
            bg_color = '#d5f4e6' if prediction_class == 'UAV' else '#fadbd8'
            
            rect = plt.Rectangle((0, 0), 1, 1, facecolor=bg_color, edgecolor=color, 
                                linewidth=3, transform=ax_pred.transAxes)
            ax_pred.add_patch(rect)
            
            ax_pred.text(0.5, 0.65, 'PREDICTED CLASS', 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        transform=ax_pred.transAxes)
            ax_pred.text(0.5, 0.35, prediction_class, 
                        ha='center', va='center', fontsize=28, fontweight='bold',
                        color=color, transform=ax_pred.transAxes)
            
            # Confidence bar (bottom right)
            ax_conf = fig.add_subplot(gs[1, 1])
            
            # Create horizontal bar
            confidence_pct = confidence * 100
            bars = ax_conf.barh(['Confidence'], [confidence_pct], 
                               color=color, height=0.5, edgecolor='black', linewidth=2)
            
            ax_conf.set_xlim([0, 100])
            ax_conf.set_xlabel('Confidence (%)', fontsize=10, fontweight='bold')
            ax_conf.set_title('Model Confidence', fontsize=11, fontweight='bold')
            
            # Add percentage text
            ax_conf.text(confidence_pct + 2, 0, f'{confidence_pct:.1f}%', 
                        va='center', fontsize=11, fontweight='bold')
            
            # Overall title
            if title is None:
                title = f'{prediction_class} Detection - {confidence:.2%} Confidence'
            
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            
            # Save
            if save_path is None:
                save_path = self.output_dir / f'prediction_{prediction_class}_{confidence:.2f}.png'
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved prediction visualization: {save_path}")
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error in visualize_single_prediction: {e}")
            plt.close()
            return None
    
    def visualize_batch_predictions(self, image_paths, predictions, confidences, 
                                   save_path=None, title=None):
        """
        Display grid of input spectrograms with predictions.
        
        Args:
            image_paths (list): List of image paths
            predictions (list): List of predicted classes
            confidences (list): List of confidence scores
            save_path (str): Path to save visualization
            title (str): Custom title
        """
        try:
            n_samples = len(image_paths)
            grid_size = int(np.ceil(np.sqrt(n_samples)))
            
            fig, axes = plt.subplots(grid_size, grid_size, 
                                    figsize=(15, 15), squeeze=False)
            fig.suptitle(title or 'Batch Prediction Analysis', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            for idx, (img_path, pred, conf) in enumerate(zip(image_paths, 
                                                             predictions, 
                                                             confidences)):
                row = idx // grid_size
                col = idx % grid_size
                ax = axes[row, col]
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(np.array(img))
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")
                
                # Color-coded border
                color = '#2ecc71' if pred == 'UAV' else '#e74c3c'
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                
                # Title with prediction and confidence
                ax.set_title(f'{pred}\n{conf:.1%} confidence', 
                            fontsize=10, fontweight='bold', color=color)
                ax.axis('off')
            
            # Hide extra subplots
            for idx in range(n_samples, grid_size * grid_size):
                row = idx // grid_size
                col = idx % grid_size
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / 'batch_predictions.png'
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved batch visualization: {save_path}")
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error in visualize_batch_predictions: {e}")
            plt.close()
            return None
    
    def compare_spectrograms(self, uav_images, bird_images, 
                            save_path=None, max_samples=5):
        """
        Compare typical UAV and Bird spectrograms side-by-side.
        
        Args:
            uav_images (list): List of UAV spectrogram paths
            bird_images (list): List of Bird spectrogram paths
            save_path (str): Path to save visualization
            max_samples (int): Maximum samples to display per class
        """
        try:
            n_samples = min(len(uav_images), len(bird_images), max_samples)
            
            fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4*n_samples))
            
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Spectrogram Characteristics Comparison\nUAV vs Bird', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            # UAV column
            for idx in range(n_samples):
                ax = axes[idx, 0]
                try:
                    img = Image.open(uav_images[idx]).convert('RGB')
                    ax.imshow(np.array(img))
                except Exception as e:
                    logger.warning(f"Could not load UAV image: {e}")
                
                ax.set_title(f'UAV Pattern {idx+1}', fontsize=11, fontweight='bold', 
                           color='#2ecc71')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2ecc71')
                    spine.set_linewidth(2)
                ax.axis('off')
            
            # Bird column
            for idx in range(n_samples):
                ax = axes[idx, 1]
                try:
                    img = Image.open(bird_images[idx]).convert('RGB')
                    ax.imshow(np.array(img))
                except Exception as e:
                    logger.warning(f"Could not load Bird image: {e}")
                
                ax.set_title(f'Bird Pattern {idx+1}', fontsize=11, fontweight='bold', 
                           color='#e74c3c')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e74c3c')
                    spine.set_linewidth(2)
                ax.axis('off')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / 'spectrogram_comparison.png'
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison visualization: {save_path}")
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error in compare_spectrograms: {e}")
            plt.close()
            return None
    
    def create_prediction_dashboard(self, image_path, prediction_class, confidence,
                                   model_metrics=None, save_path=None):
        """
        Create comprehensive prediction dashboard with model info.
        
        Args:
            image_path (str): Input spectrogram path
            prediction_class (str): Predicted class
            confidence (float): Confidence score
            model_metrics (dict): Model evaluation metrics
            save_path (str): Path to save
        """
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
            
            # Large spectrogram (left side)
            ax_img = fig.add_subplot(gs[:, 0:2])
            img = Image.open(image_path).convert('RGB')
            ax_img.imshow(np.array(img))
            ax_img.set_title('Input: Radar Micro-Doppler Spectrogram', 
                           fontsize=12, fontweight='bold')
            ax_img.axis('off')
            
            # Prediction panel
            ax_pred = fig.add_subplot(gs[0, 2])
            color = '#2ecc71' if prediction_class == 'UAV' else '#e74c3c'
            ax_pred.text(0.5, 0.7, prediction_class, ha='center', va='center',
                        fontsize=24, fontweight='bold', color=color,
                        transform=ax_pred.transAxes)
            ax_pred.text(0.5, 0.3, 'PREDICTED', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        transform=ax_pred.transAxes)
            ax_pred.set_xlim(0, 1)
            ax_pred.set_ylim(0, 1)
            ax_pred.axis('off')
            
            # Confidence
            ax_conf = fig.add_subplot(gs[1, 2])
            conf_pct = confidence * 100
            ax_conf.barh(['Confidence'], [conf_pct], color=color, edgecolor='black', linewidth=2)
            ax_conf.set_xlim([0, 100])
            ax_conf.text(conf_pct/2, 0, f'{conf_pct:.1f}%', 
                        ha='center', va='center', color='white', fontweight='bold')
            ax_conf.set_xticks([0, 25, 50, 75, 100])
            ax_conf.set_title('Model Confidence', fontsize=10, fontweight='bold')
            
            # Model metrics (if provided)
            if model_metrics:
                ax_metrics = fig.add_subplot(gs[2, 2])
                ax_metrics.axis('off')
                
                metrics_text = "Model Performance:\n"
                for key, value in model_metrics.items():
                    metrics_text += f"{key}: {value:.2f}\n"
                
                ax_metrics.text(0.1, 0.9, metrics_text, ha='left', va='top',
                              fontsize=9, family='monospace', fontweight='bold',
                              transform=ax_metrics.transAxes)
            
            fig.suptitle(f'Prediction Dashboard - {prediction_class} Detected', 
                        fontsize=14, fontweight='bold')
            
            if save_path is None:
                save_path = self.output_dir / 'prediction_dashboard.png'
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved dashboard: {save_path}")
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error in create_prediction_dashboard: {e}")
            plt.close()
            return None
