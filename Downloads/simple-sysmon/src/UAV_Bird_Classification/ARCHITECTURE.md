"""
SYSTEM ARCHITECTURE DOCUMENTATION
==================================
Comprehensive technical documentation of the UAV/Bird Classification System.
"""

# ============================================================================
# 1. SYSTEM OVERVIEW
# ============================================================================

"""
The UAV vs Bird Classification System is a complete machine learning pipeline
for distinguishing between UAVs and birds using radar micro-Doppler signatures.

Key Components:
1. Data Management (Dataset Loading)
2. Data Processing (Preprocessing & Augmentation)
3. Feature Extraction (Spectrogram Generation)
4. Model Training (CNN Architecture & Training)
5. Model Evaluation (Metrics & Visualization)
6. Inference (Prediction Module)
7. Result Storage (Database Management)
8. Report Generation (CSV/Excel/Text Reports)

Design Principles:
- Modularity: Each component is independent and reusable
- Readability: Code is well-commented and easy to understand
- Scalability: Can handle different dataset sizes and configurations
- Academic-Friendly: Simple architectures and clear explanations
"""

# ============================================================================
# 2. DATA FLOW DIAGRAM
# ============================================================================

"""
┌──────────────────────────────────────────────────────────────────────────┐
│                        INPUT DATA SOURCES                               │
├──────────────────────────────────────────────────────────────────────────┤
│  • DIAT-μSAT Dataset (Pre-generated Spectrograms)                        │
│  • Raw Radar Signals (for STFT-based Spectrogram Generation)            │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    DATASET LOADER MODULE                                │
├──────────────────────────────────────────────────────────────────────────┤
│  Functions:                                                              │
│  • load_dataset(): Load images from disk                                │
│  • get_train_test_split(): Split into train/test sets                   │
│  • get_dataset_info(): Return dataset statistics                        │
│                                                                          │
│  Output: NumPy arrays of images and corresponding labels                │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│               PREPROCESSING & AUGMENTATION MODULE                       │
├──────────────────────────────────────────────────────────────────────────┤
│  ImagePreprocessor:                                                      │
│  • resize_image()              - Resize to standard dimensions          │
│  • normalize_image()           - Min-max or z-score normalization       │
│  • reduce_noise()              - Gaussian/bilateral/morphological filter │
│  • enhance_contrast()          - CLAHE or histogram equalization        │
│  • preprocess_batch()          - Apply all operations to batch          │
│                                                                          │
│  DataAugmentation:                                                       │
│  • rotate_image()              - Random rotation                        │
│  • flip_image()                - Random flipping                        │
│  • add_gaussian_noise()        - Noise injection                        │
│  • adjust_intensity()          - Brightness adjustment                  │
│  • augment_dataset()           - Generate augmented copies              │
│                                                                          │
│  Output: Preprocessed and augmented image arrays                        │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              SPECTROGRAM GENERATION MODULE (Optional)                   │
├──────────────────────────────────────────────────────────────────────────┤
│  SpectrogramGenerator:                                                   │
│  • compute_stft()              - Compute STFT from raw signal           │
│  • generate_spectrogram_image()- Create and visualize spectrogram       │
│  • preprocess_signal()         - Remove DC, normalize                   │
│  • apply_window()              - Apply window function                  │
│  • extract_micro_doppler_signature() - Extract key features            │
│                                                                          │
│  Synthetic Signal Generation:                                            │
│  • generate_synthetic_uav_signal()  - Multi-harmonic UAV signal         │
│  • generate_synthetic_bird_signal() - Flapping bird signal              │
│                                                                          │
│  Output: STFT matrices, spectrograms, visualization plots              │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌────────────────────────┐      ┌────────────────────────┐
│   TRAIN/VAL SPLIT      │      │    TEST SET            │
│   (80% × 80%)          │      │    (20%)               │
└────────────────┬───────┘      └────────────┬───────────┘
                 │                           │
                 │                           │
                 ▼                           │
┌──────────────────────────────────────────┐ │
│        MODEL TRAINING PIPELINE           │ │
├──────────────────────────────────────────┤ │
│  1. Build CNN Architecture               │ │
│  2. Compile with Optimizer/Loss          │ │
│  3. Train with Callbacks:                │ │
│     • EarlyStopping                      │ │
│     • ReduceLROnPlateau                  │ │
│     • ModelCheckpoint                    │ │
│  4. Track Training History               │ │
│                                          │ │
│  Output: Trained model, history metrics  │ │
└────────────────────┬──────────────────────┘ │
                     │                        │
                     ▼                        ▼
             ┌──────────────┐       ┌──────────────────┐
             │ TRAINED MODEL│       │   EVALUATION SET │
             └──────────────┘       └────────┬─────────┘
                     │                       │
                     ├───────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              MODEL EVALUATION MODULE                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  ModelEvaluator:                                                         │
│  • evaluate()              - Compute all metrics                         │
│  • get_confusion_matrix()  - Classification matrix                       │
│  • get_classification_report() - Detailed per-class metrics              │
│  • plot_confusion_matrix() - Visualization                               │
│  • plot_roc_curve()        - ROC curve with AUC                          │
│  • plot_metrics()          - Metrics bar chart                           │
│                                                                          │
│  Metrics Computed:                                                       │
│  • Accuracy, Precision, Recall, F1-score, ROC-AUC                       │
│                                                                          │
│  Output: Evaluation metrics, visualizations                             │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│           PREDICTION & INFERENCE MODULE                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  ImagePredictor:                                                         │
│  • preprocess_image()      - Load and preprocess single image           │
│  • predict_single()        - Predict class for one image                │
│  • predict_batch()         - Batch predictions                          │
│  • predict_from_array()    - Predict from numpy array                   │
│  • predict_from_directory()- Predict all images in directory            │
│                                                                          │
│  PredictionAnalyzer:                                                     │
│  • get_statistics()        - Compute prediction statistics              │
│  • print_statistics()      - Display statistics                         │
│                                                                          │
│  Output: Predictions with confidence scores                             │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              DATABASE MANAGEMENT MODULE                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  PredictionDatabase:                                                     │
│  • store_prediction()      - Store single prediction                    │
│  • store_batch_predictions()- Store multiple predictions                │
│  • get_prediction()        - Retrieve prediction by ID                  │
│  • get_all_predictions()   - Get all predictions                        │
│  • get_statistics()        - Database statistics                        │
│  • export_to_csv()         - Export to CSV format                       │
│  • export_to_excel()       - Export to Excel format                     │
│                                                                          │
│  Database Schema:                                                        │
│  • predictions table: id, timestamp, image_path, class, confidence,     │
│                       probabilities, actual_class, correct, notes       │
│  • metadata table: key, value pairs for system info                     │
│                                                                          │
│  Output: Stored results, exported reports                               │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              REPORT GENERATION MODULE                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  ReportGenerator:                                                        │
│  • generate_training_report()  - Training metrics to CSV                │
│  • generate_evaluation_report()- Evaluation metrics to CSV              │
│  • generate_prediction_report()- Predictions to CSV                     │
│  • generate_summary_report()   - Comprehensive TXT summary              │
│  • generate_statistics_report()- Statistics to CSV                      │
│  • generate_excel_report()     - Multi-sheet XLSX format                │
│                                                                          │
│  ReportPrinter:                                                          │
│  • print_model_summary()   - Console output of model info               │
│  • print_dataset_summary() - Console output of dataset info             │
│                                                                          │
│  Output: CSV, Excel, and Text reports in ./reports/                     │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT & REPORTS                               │
├──────────────────────────────────────────────────────────────────────────┤
│  • training_report.csv         - Epoch-wise metrics                     │
│  • evaluation_report.csv       - Model performance metrics              │
│  • predictions_report.csv      - Per-image predictions                  │
│  • statistics_report.csv       - Prediction statistics                  │
│  • summary_report.txt          - Comprehensive summary                  │
│  • confusion_matrix.png        - Confusion matrix heatmap               │
│  • roc_curve.png               - ROC curve visualization                │
│  • metrics.png                 - Metrics bar chart                      │
│  • training_history.png        - Training curves                        │
│  • predictions.db              - SQLite database with all results       │
└──────────────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# 3. MODULE DEPENDENCY GRAPH
# ============================================================================

"""
main.py
├── dataset_loader.py
│   └── numpy, cv2, scikit-learn
├── preprocessing.py
│   └── numpy, cv2, scipy
├── spectrogram.py
│   └── numpy, scipy, matplotlib
├── model.py
│   └── tensorflow, keras
├── train.py
│   ├── tensorflow, keras
│   └── evaluation.py
├── evaluate.py
│   ├── numpy, tensorflow, sklearn, matplotlib, seaborn
│   └── (uses model.py output)
├── predict.py
│   ├── numpy, tensorflow, cv2
│   └── (uses model.py output)
├── database.py
│   ├── sqlite3, pandas, json
│   └── (stores predict.py output)
└── report.py
    ├── pandas, numpy, matplotlib
    └── (uses all module outputs)
"""

# ============================================================================
# 4. CLASS RELATIONSHIPS
# ============================================================================

"""
UAVBirdClassificationSystem (Main Orchestrator)
    ├── Composes: DatasetLoader
    ├── Composes: ImagePreprocessor
    ├── Composes: SpectrogramGenerator
    ├── Composes: UAVBirdCNN
    ├── Composes: ModelTrainer
    ├── Composes: ModelEvaluator
    ├── Composes: ImagePredictor
    ├── Composes: PredictionDatabase
    └── Composes: ReportGenerator

Key Design Patterns:
- Factory Pattern: UAVBirdCNN.build_model() creates model instances
- Strategy Pattern: Different preprocessing strategies (normalize, denoise, etc.)
- Observer Pattern: Training callbacks (EarlyStopping, ModelCheckpoint)
- Repository Pattern: PredictionDatabase abstracts data persistence
"""

# ============================================================================
# 5. CNN ARCHITECTURE DETAILS
# ============================================================================

"""
Input: 128×128×1 (Grayscale spectrogram)

Block 1 (Extraction):
  Conv2D(32, 3×3) → ReLU
  Conv2D(32, 3×3) → ReLU
  MaxPool2D(2×2) → 64×64×32
  BatchNorm → Normalized activations

Block 2 (Refinement):
  Conv2D(64, 3×3) → ReLU
  Conv2D(64, 3×3) → ReLU
  MaxPool2D(2×2) → 32×32×64
  BatchNorm → Normalized activations

Block 3 (Advanced Features):
  Conv2D(128, 3×3) → ReLU
  Conv2D(128, 3×3) → ReLU
  MaxPool2D(2×2) → 16×16×128
  BatchNorm → Normalized activations

Global Aggregation:
  GlobalAveragePooling2D → 128 values

Dense Layers (Classification):
  Dense(256) → ReLU → Dropout(0.5)
  Dense(128) → ReLU → Dropout(0.5)
  Dense(64) → ReLU → Dropout(0.3)
  
Output:
  Dense(2) → Softmax → [P(UAV), P(Bird)]

Total Parameters: ~1.2M (suitable for academic projects)
Trainable Parameters: All

Key Features:
- Batch Normalization: Stabilizes training, allows higher learning rates
- Dropout: Regularization to prevent overfitting
- Global Average Pooling: Reduces spatial dimensions, robust to spatial shifts
- ReLU Activation: Non-linearity for feature learning
- Softmax Output: Probability distribution over classes
"""

# ============================================================================
# 6. TRAINING PIPELINE
# ============================================================================

"""
Data Preparation:
  1. Load dataset (1000+ images, 128×128 resolution)
  2. Split: 80% train, 20% test
  3. Further split train: 80% train, 20% validation
  4. Normalize to [0, 1]
  5. Add channel dimension: (N, 128, 128) → (N, 128, 128, 1)
  6. One-hot encode labels: [0, 1] → [[1, 0], [0, 1]]

Model Compilation:
  Optimizer: Adam (adaptive learning rate)
    - Initial LR: 0.001
    - β₁: 0.9, β₂: 0.999, ε: 1e-7
  
  Loss: Categorical Crossentropy
    - log loss for multi-class classification
  
  Metrics: Accuracy
    - Proportion of correct predictions

Training Loop:
  for epoch in range(max_epochs):
    for batch in training_data:
      1. Forward pass: predictions = model(batch)
      2. Compute loss: loss = categorical_crossentropy(targets, predictions)
      3. Backward pass: compute gradients
      4. Update weights: optimizer.update(gradients)
    
    Validation:
      1. Evaluate on validation set
      2. Track metrics (accuracy, loss)
    
    Callbacks:
      - EarlyStopping: Stop if val_loss doesn't improve for 10 epochs
      - ReduceLROnPlateau: Reduce LR by 0.5 if plateau detected
      - ModelCheckpoint: Save best model based on val_accuracy

Output: Trained weights, training history (100+ metrics)
"""

# ============================================================================
# 7. EVALUATION METHODOLOGY
# ============================================================================

"""
Test Set Evaluation:
  1. Load trained model
  2. Make predictions on test set
  3. Compare predictions with ground truth

Metrics Computed:

  Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Overall correctness

  Precision = TP / (TP + FP)
    - How many predicted positive are actually positive

  Recall = TP / (TP + FN)
    - How many actual positives are correctly identified

  F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
    - Harmonic mean of precision and recall

  ROC-AUC = Area under ROC curve
    - Measure of separability between classes
    - 0.5 = random, 1.0 = perfect

Confusion Matrix:
  [[TP_UAV,  FN_UAV],
   [FP_Bird, TP_Bird]]

  TP: True Positives (correctly classified as class)
  TN: True Negatives (correctly rejected)
  FP: False Positives (incorrectly classified as class)
  FN: False Negatives (incorrectly rejected)

Visual Outputs:
  - Confusion matrix heatmap
  - ROC curve with AUC score
  - Metrics bar chart
  - Training history curves (accuracy, loss)
"""

# ============================================================================
# 8. PREDICTION PIPELINE
# ============================================================================

"""
Single Image Prediction:

  1. Load image from disk
     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  
  2. Preprocess image
     - Resize: (H, W) → (128, 128)
     - Normalize: values / 255.0 → [0, 1]
     - Add channel: (128, 128) → (128, 128, 1)
  
  3. Add batch dimension
     image_batch = expand_dims(image, axis=0) → (1, 128, 128, 1)
  
  4. Forward pass
     probabilities = model.predict(image_batch)
     → [P(UAV), P(Bird)]
  
  5. Predict class
     predicted_class_idx = argmax(probabilities)
     predicted_class = class_names[predicted_class_idx]
  
  6. Get confidence
     confidence = max(probabilities)
  
  7. Return result
     {
       'class': 'UAV' or 'Bird',
       'confidence': 0.0-1.0,
       'probabilities': {'UAV': float, 'Bird': float},
       'image_path': str,
       'meets_threshold': boolean
     }

Batch Prediction:
  - Apply single prediction logic to multiple images
  - Process in batches for efficiency
  - Aggregate results

Directory Prediction:
  - Find all images in directory
  - Predict on all images
  - Organize results by class
"""

# ============================================================================
# 9. DATABASE SCHEMA
# ============================================================================

"""
SQLite Database: predictions.db

Table: predictions
┌──────────────┬──────────┬────────────────────────────────┐
│ Column       │ Type     │ Purpose                        │
├──────────────┼──────────┼────────────────────────────────┤
│ id           │ INTEGER  │ Primary key, auto-increment    │
│ timestamp    │ DATETIME │ When prediction was made       │
│ image_path   │ TEXT     │ Path to image file             │
│ pred_class   │ TEXT     │ Predicted class (UAV/Bird)     │
│ confidence   │ REAL     │ Confidence score [0, 1]        │
│ probabilities│ TEXT     │ JSON of all probabilities      │
│ actual_class │ TEXT     │ Ground truth (if known)        │
│ correct      │ BOOLEAN  │ Whether prediction was correct │
│ notes        │ TEXT     │ Optional additional info       │
└──────────────┴──────────┴────────────────────────────────┘

Table: metadata
┌──────────────┬──────────┬────────────────────────────────┐
│ Column       │ Type     │ Purpose                        │
├──────────────┼──────────┼────────────────────────────────┤
│ id           │ INTEGER  │ Primary key, auto-increment    │
│ key          │ TEXT     │ Metadata key (unique)          │
│ value        │ TEXT     │ Metadata value                 │
│ timestamp    │ DATETIME │ When metadata was stored       │
└──────────────┴──────────┴────────────────────────────────┘

Typical Metadata Entries:
  - 'model_version': '1.0'
  - 'dataset_name': 'DIAT-μSAT'
  - 'total_samples': '1000'
  - 'training_date': '2026-02-10'
  - 'final_accuracy': '0.89'
"""

# ============================================================================
# 10. REPORT FORMATS
# ============================================================================

"""
CSV Reports Format:

training_report.csv:
  Epoch,accuracy,val_accuracy,loss,val_loss
  1,0.650,0.620,0.564,0.598
  2,0.720,0.680,0.423,0.456
  ...

evaluation_report.csv:
  Metric,Value
  Accuracy,0.8900
  Precision,0.8750
  Recall,0.9050
  F1-score,0.8900
  Roc Auc,0.9234

predictions_report.csv:
  Image_Path,Predicted_Class,Confidence,UAV_Probability,Bird_Probability
  /path/img1.png,UAV,0.9523,0.9523,0.0477
  /path/img2.png,Bird,0.8732,0.1268,0.8732
  ...

Text Report Format:

summary_report.txt:
  ============================================================
  UAV vs BIRD CLASSIFICATION SYSTEM - SUMMARY REPORT
  ============================================================
  Generated: 2026-02-10 14:30:45
  
  MODEL INFORMATION
  Architecture: CNN
  Input Shape: (128, 128, 1)
  Total Parameters: 1234567
  
  DATASET INFORMATION
  Total Samples: 1000
  UAV Samples: 500
  Bird Samples: 500
  Train/Test Split: 80/20
  
  TRAINING METRICS
  Epochs Trained: 50
  Final Training Accuracy: 0.9234
  Final Validation Accuracy: 0.9012
  
  EVALUATION METRICS
  Accuracy: 0.8900
  Precision: 0.8750
  Recall: 0.9050
  F1-score: 0.8900
  Roc Auc: 0.9234
  ============================================================
"""

# ============================================================================
# 11. CONFIGURATION PARAMETERS
# ============================================================================

"""
Default Configuration Dictionary:

{
  'dataset_path': './dataset',           # Location of dataset
  'image_size': (128, 128),              # Input image dimensions
  'batch_size': 32,                      # Training batch size
  'epochs': 50,                          # Max training epochs
  'test_split': 0.2,                     # 20% for testing
  'validation_split': 0.2,               # 20% of train for validation
  'learning_rate': 0.001,                # Adam optimizer LR
  'model_save_path': './model/trained_model.h5',  # Model save location
  'db_path': './database/predictions.db',         # Database location
  'reports_dir': './reports',                     # Reports output dir
  'augment_data': True,                  # Enable data augmentation
  'augment_factor': 2,                   # Augmentation multiplier
  'verbose': 1                           # Training verbosity
}

Customization:
  system = UAVBirdClassificationSystem(config=custom_config)
"""

# ============================================================================
# 12. PERFORMANCE CHARACTERISTICS
# ============================================================================

"""
Computational Complexity:

Model Training:
  - Time: ~10-30 minutes (depending on hardware, dataset size)
  - Memory: ~4-8 GB GPU VRAM recommended
  - Forward pass: O(n) where n = batch size
  - Backward pass: O(n)

Model Inference (Single Image):
  - Time: ~10-50 ms per image
  - Memory: ~100-500 MB
  - Speed depends on hardware

Typical Accuracy Ranges:
  - Training Accuracy: 85-95%
  - Validation Accuracy: 80-90%
  - Test Accuracy: 80-90%
  - ROC-AUC: 0.85-0.95

Memory Requirements:
  - Model weights: ~5 MB
  - Single image: ~128 KB (preprocessed)
  - Batch of 32: ~4 MB
  - Database growth: ~1 KB per prediction

Scalability:
  - Can handle 1000s of predictions
  - Database grows linearly with predictions
  - Training time scales with dataset size
"""

# ============================================================================
# 13. EXTENDING THE SYSTEM
# ============================================================================

"""
Common Extensions:

1. Modify CNN Architecture:
   - Edit model.py UAVBirdCNN class
   - Add more conv blocks, adjust filter numbers
   - Implement ResNet or other architectures

2. Add New Preprocessing Technique:
   - Create new method in ImagePreprocessor class
   - Add to preprocess_batch() pipeline

3. Implement Transfer Learning:
   - Use build_transfer_learning_model() in model.py
   - Fine-tune pre-trained models (ResNet, VGG, etc.)

4. Add New Evaluation Metric:
   - Create new method in ModelEvaluator class
   - Compute and store metric

5. Custom Data Augmentation:
   - Extend DataAugmentation class
   - Implement domain-specific augmentation

6. Multi-Class Classification:
   - Modify dataset structure (3+ classes)
   - Change Dense output layer units
   - Update evaluation metrics for multi-class

7. Real-Time Prediction:
   - Use prediction module on incoming streams
   - Implement buffering/queueing
   - Add real-time visualization

8. Web Interface:
   - Add Flask/Django backend
   - Create REST API endpoints
   - Build HTML/React frontend
"""

# ============================================================================
# 14. TROUBLESHOOTING GUIDE
# ============================================================================

"""
Common Issues and Solutions:

Issue: "Module not found" Error
Solution: 
  - Ensure all files are in correct directories
  - Check Python path includes project directory
  - Import statements match file structure

Issue: "Out of Memory" Error
Solution:
  - Reduce batch_size (e.g., 16 instead of 32)
  - Reduce image_size (e.g., 64 instead of 128)
  - Use CPU mode: pip install tensorflow-cpu
  - Process data in smaller chunks

Issue: "Dataset not found" Error
Solution:
  - Verify dataset directory exists: ./dataset/
  - Check subdirectories: ./dataset/UAV/, ./dataset/Bird/
  - Verify image formats supported (.png, .jpg, .jpeg, .bmp)

Issue: "Model training stalled"
Solution:
  - Check learning rate (try 0.0001 or 0.01)
  - Increase batch normalization momentum
  - Verify data is properly normalized
  - Add data augmentation

Issue: "Low accuracy"
Solution:
  - Check data quality and labels
  - Increase training epochs
  - Add more data/augmentation
  - Modify architecture complexity
  - Check class imbalance

Issue: "CUDA/GPU not working"
Solution:
  - Check NVIDIA drivers: nvidia-smi
  - Install tensorflow-gpu: pip install tensorflow-gpu
  - Verify CUDA toolkit compatibility
  - Use CPU version if problems persist

Issue: "Reports not generated"
Solution:
  - Ensure ./reports/ directory exists
  - Check write permissions
  - Verify report_path in config
  - Run report.py individually to debug
"""

# ============================================================================
# 15. FUTURE IMPROVEMENTS
# ============================================================================

"""
Potential Enhancements:

1. Advanced Architectures:
   - Attention mechanisms (Transformer blocks)
   - Residual connections (ResNet)
   - Inception modules
   - Mobile-friendly architectures (MobileNet)

2. Improved Data Processing:
   - Synthetic spectrogram generation via GANs
   - 3D spectrograms (time, frequency, range)
   - Mel-frequency spectrograms
   - Wavelet analysis

3. Ensemble Methods:
   - Multiple models voting
   - Stacking and blending
   - Soft voting with confidence scores

4. Real-Time Processing:
   - Streaming signal processing
   - Live prediction pipeline
   - Edge deployment (TensorFlow Lite)

5. Explainability:
   - Grad-CAM visualization
   - SHAP values
   - Feature importance analysis
   - Decision boundary visualization

6. Multi-Task Learning:
   - Joint classification and localization
   - Speed estimation
   - Altitude estimation

7. Deployment Options:
   - REST API (Flask/FastAPI)
   - Web interface (React/Vue)
   - Mobile app (TensorFlow Lite)
   - Embedded systems (NVIDIA Jetson)

8. Continuous Learning:
   - Online learning from new predictions
   - Model retraining pipeline
   - Active learning strategies

9. Data Augmentation:
   - Frequency domain augmentation
   - Time warping
   - Mixup and CutMix
   - AutoAugment

10. Optimization:
    - Quantization (INT8)
    - Pruning (weight reduction)
    - Knowledge distillation
    - Model compression
"""

print("""
═══════════════════════════════════════════════════════════════════════════════
                    SYSTEM ARCHITECTURE DOCUMENTATION
                              COMPLETE OVERVIEW

The UAV vs Bird Classification System is a comprehensive, modular, and
extensible platform for deep learning-based radar target classification.

Key Characteristics:
✓ Modular Design: Independent, reusable components
✓ Academic-Friendly: Simple, explainable architectures
✓ Production-Ready: Comprehensive error handling and logging
✓ Well-Documented: Detailed docstrings and examples
✓ Scalable: Handles various dataset sizes and configurations
✓ Flexible: Easy to extend and customize

For detailed information on each module, see the module docstrings.
For quick start guide, see QUICKSTART.md
For user guide, see README.md

═══════════════════════════════════════════════════════════════════════════════
""")
