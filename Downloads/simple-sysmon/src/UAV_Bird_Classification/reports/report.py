"""
Report Generation Module
========================
This module generates comprehensive reports of model performance
and predictions in CSV and Excel formats.

Author: B.Tech Major Project
Date: 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive reports for model training and evaluation.
    
    Report types:
    - Training Report: Model training history
    - Evaluation Report: Model performance metrics
    - Prediction Report: Detailed predictions
    - Summary Report: Overall system summary
    """
    
    def __init__(self, output_dir='./reports'):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir (str): Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ReportGenerator with output directory: {output_dir}")
    
    def generate_training_report(self, history, output_filename='training_report.csv'):
        """
        Generate training history report.
        
        Args:
            history (dict): Training history from model.fit()
            output_filename (str): Output CSV filename
        
        Returns:
            str: Path to saved report
        """
        logger.info("Generating training report...")
        
        # Create DataFrame from history
        df = pd.DataFrame(history)
        
        # Add epoch column
        df.insert(0, 'Epoch', range(1, len(df) + 1))
        
        # Save to CSV
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Training report saved to {output_path}")
        
        return str(output_path)
    
    def generate_evaluation_report(self, metrics, confusion_matrix=None,
                                   output_filename='evaluation_report.csv'):
        """
        Generate evaluation metrics report.
        
        Args:
            metrics (dict): Evaluation metrics
            confusion_matrix (np.ndarray): Confusion matrix
            output_filename (str): Output CSV filename
        
        Returns:
            str: Path to saved report
        """
        logger.info("Generating evaluation report...")
        
        report_data = {
            'Metric': [],
            'Value': []
        }
        
        # Add metrics
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                report_data['Metric'].append(metric_name.replace('_', ' ').title())
                report_data['Value'].append(metric_value)
        
        # Create DataFrame
        df = pd.DataFrame(report_data)
        
        # Save to CSV
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Also save confusion matrix if provided
        if confusion_matrix is not None:
            cm_path = self.output_dir / 'confusion_matrix.csv'
            cm_df = pd.DataFrame(
                confusion_matrix,
                index=['True UAV', 'True Bird'],
                columns=['Predicted UAV', 'Predicted Bird']
            )
            cm_df.to_csv(cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")
        
        return str(output_path)
    
    def generate_prediction_report(self, predictions, output_filename='predictions_report.csv'):
        """
        Generate detailed prediction report.
        
        Args:
            predictions (list): List of prediction dictionaries
            output_filename (str): Output CSV filename
        
        Returns:
            str: Path to saved report
        """
        logger.info("Generating prediction report...")
        
        # Prepare data for DataFrame
        report_data = {
            'Image_Path': [],
            'Predicted_Class': [],
            'Confidence': [],
            'UAV_Probability': [],
            'Bird_Probability': [],
            'Timestamp': []
        }
        
        for pred in predictions:
            report_data['Image_Path'].append(pred.get('image_path', ''))
            report_data['Predicted_Class'].append(pred.get('class', ''))
            report_data['Confidence'].append(pred.get('confidence', 0.0))
            
            probs = pred.get('probabilities', {})
            report_data['UAV_Probability'].append(probs.get('UAV', 0.0))
            report_data['Bird_Probability'].append(probs.get('Bird', 0.0))
            report_data['Timestamp'].append(datetime.now().isoformat())
        
        # Create DataFrame
        df = pd.DataFrame(report_data)
        
        # Save to CSV
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Prediction report saved to {output_path}")
        
        return str(output_path)
    
    def generate_summary_report(self, model_info=None, dataset_info=None,
                                training_metrics=None, evaluation_metrics=None,
                                output_filename='summary_report.txt'):
        """
        Generate comprehensive summary report.
        
        Args:
            model_info (dict): Model architecture information
            dataset_info (dict): Dataset statistics
            training_metrics (dict): Training metrics
            evaluation_metrics (dict): Evaluation metrics
            output_filename (str): Output filename
        
        Returns:
            str: Path to saved report
        """
        logger.info("Generating summary report...")
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 70)
        report_lines.append("UAV vs BIRD CLASSIFICATION SYSTEM - SUMMARY REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Model Information
        if model_info:
            report_lines.append("MODEL INFORMATION")
            report_lines.append("-" * 70)
            for key, value in model_info.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Dataset Information
        if dataset_info:
            report_lines.append("DATASET INFORMATION")
            report_lines.append("-" * 70)
            for key, value in dataset_info.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Training Metrics
        if training_metrics:
            report_lines.append("TRAINING METRICS")
            report_lines.append("-" * 70)
            for key, value in training_metrics.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Evaluation Metrics
        if evaluation_metrics:
            report_lines.append("EVALUATION METRICS")
            report_lines.append("-" * 70)
            for key, value in evaluation_metrics.items():
                if value is not None:
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)
        
        # Write to file
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to {output_path}")
        
        return str(output_path)
    
    def generate_excel_report(self, data_dict, output_filename='comprehensive_report.xlsx'):
        """
        Generate comprehensive Excel report with multiple sheets.
        
        Args:
            data_dict (dict): Dictionary of sheet_name -> data
            output_filename (str): Output filename
        
        Returns:
            str: Path to saved report
        """
        logger.info("Generating Excel report...")
        
        output_path = self.output_dir / output_filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, data in data_dict.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(data, dict):
                    df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Excel report saved to {output_path}")
        
        return str(output_path)
    
    def generate_statistics_report(self, predictions, class_names=['UAV', 'Bird'],
                                   output_filename='statistics_report.csv'):
        """
        Generate detailed statistics report from predictions.
        
        Args:
            predictions (list): List of predictions
            class_names (list): Class names
            output_filename (str): Output filename
        
        Returns:
            str: Path to saved report
        """
        logger.info("Generating statistics report...")
        
        # Compute statistics
        stats = {
            'Metric': [],
            'UAV': [],
            'Bird': [],
            'Overall': []
        }
        
        # Total predictions
        total = len(predictions)
        uav_count = sum(1 for p in predictions if p.get('class') == 'UAV')
        bird_count = sum(1 for p in predictions if p.get('class') == 'Bird')
        
        stats['Metric'].append('Count')
        stats['UAV'].append(uav_count)
        stats['Bird'].append(bird_count)
        stats['Overall'].append(total)
        
        # Percentage
        stats['Metric'].append('Percentage (%)')
        stats['UAV'].append(round((uav_count / total * 100) if total > 0 else 0, 2))
        stats['Bird'].append(round((bird_count / total * 100) if total > 0 else 0, 2))
        stats['Overall'].append(100.0)
        
        # Average confidence by class
        uav_confidences = [p.get('confidence', 0) for p in predictions if p.get('class') == 'UAV']
        bird_confidences = [p.get('confidence', 0) for p in predictions if p.get('class') == 'Bird']
        all_confidences = [p.get('confidence', 0) for p in predictions]
        
        stats['Metric'].append('Avg Confidence')
        stats['UAV'].append(round(np.mean(uav_confidences) if uav_confidences else 0, 4))
        stats['Bird'].append(round(np.mean(bird_confidences) if bird_confidences else 0, 4))
        stats['Overall'].append(round(np.mean(all_confidences) if all_confidences else 0, 4))
        
        # Min confidence
        stats['Metric'].append('Min Confidence')
        stats['UAV'].append(round(np.min(uav_confidences) if uav_confidences else 0, 4))
        stats['Bird'].append(round(np.min(bird_confidences) if bird_confidences else 0, 4))
        stats['Overall'].append(round(np.min(all_confidences) if all_confidences else 0, 4))
        
        # Max confidence
        stats['Metric'].append('Max Confidence')
        stats['UAV'].append(round(np.max(uav_confidences) if uav_confidences else 0, 4))
        stats['Bird'].append(round(np.max(bird_confidences) if bird_confidences else 0, 4))
        stats['Overall'].append(round(np.max(all_confidences) if all_confidences else 0, 4))
        
        # Create DataFrame
        df = pd.DataFrame(stats)
        
        # Save to CSV
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Statistics report saved to {output_path}")
        
        return str(output_path)


class ReportPrinter:
    """
    Prints formatted reports to console.
    """
    
    @staticmethod
    def print_model_summary(model_info):
        """
        Print model information.
        
        Args:
            model_info (dict): Model information
        """
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")
    
    @staticmethod
    def print_dataset_summary(dataset_info):
        """
        Print dataset information.
        
        Args:
            dataset_info (dict): Dataset information
        """
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")


def main():
    """
    Example usage of ReportGenerator.
    """
    print("Report Generation Module Test")
    print("=" * 60)
    
    # Initialize generator
    generator = ReportGenerator('./test_reports')
    
    # Test data
    history = {
        'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'accuracy': [0.7, 0.75, 0.8, 0.85, 0.9],
        'val_loss': [0.6, 0.5, 0.4, 0.3, 0.25],
        'val_accuracy': [0.65, 0.7, 0.75, 0.8, 0.85]
    }
    
    metrics = {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1_score': 0.85,
        'roc_auc': 0.90
    }
    
    predictions = [
        {'image_path': '/path/image1.png', 'class': 'UAV', 'confidence': 0.95,
         'probabilities': {'UAV': 0.95, 'Bird': 0.05}},
        {'image_path': '/path/image2.png', 'class': 'Bird', 'confidence': 0.88,
         'probabilities': {'UAV': 0.12, 'Bird': 0.88}},
    ]
    
    # Generate reports
    print("\nGenerating reports...")
    
    training_path = generator.generate_training_report(history)
    print(f"Training report: {training_path}")
    
    eval_path = generator.generate_evaluation_report(metrics)
    print(f"Evaluation report: {eval_path}")
    
    pred_path = generator.generate_prediction_report(predictions)
    print(f"Prediction report: {pred_path}")
    
    stats_path = generator.generate_statistics_report(predictions)
    print(f"Statistics report: {stats_path}")
    
    summary_path = generator.generate_summary_report(
        model_info={'Architecture': 'CNN', 'Parameters': '1.2M'},
        dataset_info={'Total Samples': 1000, 'Train/Test Split': '80/20'},
        training_metrics={'Best Epoch': 50, 'Final Accuracy': 0.90},
        evaluation_metrics=metrics
    )
    print(f"Summary report: {summary_path}")
    
    print("\nReport generation test completed!")


if __name__ == "__main__":
    main()
