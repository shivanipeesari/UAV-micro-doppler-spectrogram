"""
Quick Start Guide
=================
Get started with the UAV Bird Classification system in 5 minutes.
"""

# ============================================================================
# QUICK START GUIDE
# ============================================================================

# 1. INSTALLATION
# ============================================================================
# pip install -r requirements.txt

# 2. BASIC USAGE
# ============================================================================

# Option A: Run complete pipeline (training + evaluation)
# --------------------------------------------------------
from main import UAVBirdClassificationSystem

system = UAVBirdClassificationSystem()
result = system.run_complete_pipeline()

# Option B: Load dataset only
# --------------------------------------------------------
from dataset.dataset_loader import DatasetLoader

loader = DatasetLoader('./dataset', image_size=(128, 128))
images, labels, paths = loader.load_dataset()
X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)

# Option C: Train model on loaded data
# --------------------------------------------------------
import numpy as np
from tensorflow.keras.utils import to_categorical
from model.model import UAVBirdCNN
from training.train import ModelTrainer

# Prepare data
y_train_oh = to_categorical(y_train, 2)
y_test_oh = to_categorical(y_test, 2)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build and train model
cnn = UAVBirdCNN(input_shape=(128, 128, 1), num_classes=2)
model = cnn.build_model()
cnn.compile_model()

trainer = ModelTrainer(model)
history = trainer.train(X_train, y_train_oh, X_test, y_test_oh, epochs=50)

# Option D: Evaluate trained model
# --------------------------------------------------------
from evaluation.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model, class_names=['UAV', 'Bird'])
metrics = evaluator.evaluate(X_test, y_test_oh)
evaluator.print_evaluation_summary()
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curve()

# Option E: Make predictions on new images
# --------------------------------------------------------
from database.predict import ImagePredictor, PredictionAnalyzer

predictor = ImagePredictor(model, class_names=['UAV', 'Bird'])

# Single image prediction
result = predictor.predict_single('./test_image.png')
print(f"Predicted: {result['class']} (Confidence: {result['confidence']:.4f})")

# Batch prediction
predictions = predictor.predict_batch(['image1.png', 'image2.png', 'image3.png'])

# Analyze predictions
analyzer = PredictionAnalyzer(predictions)
analyzer.print_statistics()

# Option F: Store predictions in database
# --------------------------------------------------------
from database.database import PredictionDatabase

db = PredictionDatabase('./database/predictions.db')
db.store_batch_predictions(predictions)

# Get statistics
stats = db.get_statistics()
print(f"Total predictions: {stats['total_predictions']}")

# Export to CSV/Excel
db.export_to_csv('./predictions.csv')
db.export_to_excel('./predictions.xlsx')

# Option G: Generate reports
# --------------------------------------------------------
from reports.report import ReportGenerator

generator = ReportGenerator('./reports')
generator.generate_training_report(history)
generator.generate_evaluation_report(metrics)
generator.generate_prediction_report(predictions)
generator.generate_summary_report(
    model_info={'Architecture': 'CNN', 'Parameters': '1.2M'},
    training_metrics={'Epochs': 50, 'Batch Size': 32},
    evaluation_metrics=metrics
)

# 3. COMMON TASKS
# ============================================================================

# Preprocess images
from preprocessing.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(128, 128))
preprocessed = preprocessor.preprocess_batch(
    images,
    normalize=True,
    denoise=True,
    enhance_contrast=True
)

# Generate spectrograms from raw signals
from spectrogram.spectrogram import SpectrogramGenerator

generator = SpectrogramGenerator(sampling_rate=1000, n_fft=512)
uav_signal = generator.generate_synthetic_uav_signal(duration=1.0)
signal_processed = generator.preprocess_signal(uav_signal)
f, t, spectrogram = generator.compute_stft(signal_processed)
generator.generate_spectrogram_image(signal_processed, output_path='spectrogram.png')

# Data augmentation
from preprocessing.preprocessing import DataAugmentation

augmentor = DataAugmentation()
aug_images, aug_labels = augmentor.augment_dataset(
    images, labels, augment_factor=2
)

# 4. CONFIGURATION
# ============================================================================

# Custom configuration
config = {
    'dataset_path': './dataset',
    'image_size': (128, 128),
    'batch_size': 32,
    'epochs': 50,
    'test_split': 0.2,
    'learning_rate': 0.001,
    'model_save_path': './model/trained_model.h5',
    'db_path': './database/predictions.db',
    'reports_dir': './reports'
}

system = UAVBirdClassificationSystem(config=config)

# 5. TESTING
# ============================================================================

# Test with dummy data
import numpy as np

# Create dummy dataset
dummy_images = np.random.rand(100, 128, 128).astype(np.float32)
dummy_labels = np.random.randint(0, 2, 100)

# Test preprocessing
preprocessor = ImagePreprocessor()
processed = preprocessor.preprocess_batch(dummy_images)

# Test model
from model.model import UAVBirdCNN
cnn = UAVBirdCNN()
model = cnn.build_model()
cnn.get_model_summary()

# 6. TROUBLESHOOTING
# ============================================================================

# Issue: "Dataset not found"
# Solution: Ensure dataset folder structure:
#   dataset/
#   ├── UAV/
#   │   ├── image1.png
#   │   └── ...
#   └── Bird/
#       ├── image1.png
#       └── ...

# Issue: "Out of Memory"
# Solution: Reduce batch size or image size
config['batch_size'] = 16
config['image_size'] = (64, 64)

# Issue: "CUDA/GPU not available"
# Solution: Install CPU version
# pip install tensorflow-cpu

# 7. NEXT STEPS
# ============================================================================

# 1. Prepare your dataset (DIAT-μSAT or custom)
# 2. Adjust configuration parameters as needed
# 3. Run the complete pipeline
# 4. Analyze results and reports
# 5. Fine-tune model if necessary
# 6. Deploy for predictions on new data

# For detailed documentation, see README.md
# For individual module documentation, see docstrings in each module

print("""
╔════════════════════════════════════════════════════════════════╗
║  UAV vs BIRD CLASSIFICATION SYSTEM - QUICK START GUIDE         ║
║                                                                ║
║  Start with: system = UAVBirdClassificationSystem()            ║
║  Then:       result = system.run_complete_pipeline()          ║
║                                                                ║
║  See README.md for detailed documentation                      ║
╚════════════════════════════════════════════════════════════════╝
""")
