"""
Visualization Module
====================
Professional visualization tools for UAV vs Bird classification system.

Includes:
- Training history plots
- Confusion matrix heatmap
- ROC and Precision-Recall curves
- Metrics comparison
- Prediction confidence distribution
- Sample predictions display
- Learning curves analysis
- Model architecture analysis
"""

from .visualizer import AdvancedVisualizer, ModelAnalyzer

__all__ = ['AdvancedVisualizer', 'ModelAnalyzer']
