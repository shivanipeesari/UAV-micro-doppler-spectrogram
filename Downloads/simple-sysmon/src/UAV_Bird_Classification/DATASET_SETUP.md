# Dataset Preparation Guide

## DIAT-μSAT Dataset Setup

This guide explains how to prepare and organize the DIAT-μSAT (Small Aerial Targets' Micro-Doppler Signatures) dataset for use with the UAV/Bird Classification System.

---

## Dataset Information

### About DIAT-μSAT
- **Full Name**: DIAT-μSAT: Small Aerial Targets' Micro-Doppler Signatures and Their Classification Using CNN
- **Source**: Research dataset (publicly available)
- **Size**: Thousands of spectrogram images
- **Classes**: UAV and Bird micro-Doppler signatures
- **Format**: Primarily PNG and JPG images
- **Resolution**: Typically 256×256 or 512×512 pixels
- **Frequency Domain**: Time-frequency spectrograms from radar signals

### Access the Dataset
1. Visit: https://www.researchgate.net/publication/353816267_DIAT-mSAT_Small_Aerial_Targets'_Micro-Doppler_Signatures_and_Their_Classification_Using_CNN
2. Request access if needed
3. Download the dataset ZIP file

---

## Directory Structure Setup

### Step 1: Extract Dataset
```bash
# After downloading DIAT-μSAT dataset
unzip DIAT-mSAT_dataset.zip -d ./raw_dataset/
```

### Step 2: Create Project Directory Structure
```bash
cd UAV_Bird_Classification

# Create required directories
mkdir -p dataset/UAV
mkdir -p dataset/Bird
```

### Step 3: Organize Images

The dataset should be organized as follows:

```
dataset/
├── UAV/
│   ├── uav_001.png
│   ├── uav_002.png
│   ├── uav_003.jpg
│   ├── uav_004.png
│   └── ... (all UAV spectrograms)
│
└── Bird/
    ├── bird_001.png
    ├── bird_002.png
    ├── bird_003.jpg
    ├── bird_004.png
    └── ... (all Bird spectrograms)
```

### Step 4: Copy Images to Project
```bash
# Copy UAV spectrograms
cp path/to/downloaded/UAVs/* dataset/UAV/

# Copy Bird spectrograms
cp path/to/downloaded/Birds/* dataset/Bird/

# Verify
ls -la dataset/UAV/ | wc -l
ls -la dataset/Bird/ | wc -l
```

---

## Alternative: If Dataset Access is Limited

### Generate Synthetic Dataset for Testing

If you don't have access to DIAT-μSAT, you can generate synthetic spectrograms:

```python
import os
import numpy as np
from PIL import Image
from spectrogram.spectrogram import SpectrogramGenerator

# Initialize spectrogram generator
generator = SpectrogramGenerator(sampling_rate=1000, n_fft=512)

# Create output directories
os.makedirs('dataset/UAV', exist_ok=True)
os.makedirs('dataset/Bird', exist_ok=True)

# Generate synthetic UAV spectrograms
print("Generating synthetic UAV spectrograms...")
for i in range(100):  # 100 UAV examples
    # Generate synthetic signal
    uav_signal = generator.generate_synthetic_uav_signal(
        duration=1.0,
        base_frequency=150 + np.random.randint(-50, 50)
    )
    
    # Preprocess signal
    processed = generator.preprocess_signal(uav_signal)
    
    # Generate spectrogram
    f, t, spec = generator.compute_stft(processed)
    
    # Convert to image (normalize to 0-255)
    spec_img = ((spec - spec.min()) / (spec.max() - spec.min()) * 255).astype(np.uint8)
    
    # Save as image
    img = Image.fromarray(spec_img)
    img.save(f'dataset/UAV/uav_{i:03d}.png')
    
    if (i + 1) % 20 == 0:
        print(f"  Generated {i + 1} UAV spectrograms")

# Generate synthetic Bird spectrograms
print("Generating synthetic Bird spectrograms...")
for i in range(100):  # 100 Bird examples
    # Generate synthetic signal
    bird_signal = generator.generate_synthetic_bird_signal(
        duration=1.0,
        base_frequency=80 + np.random.randint(-20, 20)
    )
    
    # Preprocess signal
    processed = generator.preprocess_signal(bird_signal)
    
    # Generate spectrogram
    f, t, spec = generator.compute_stft(processed)
    
    # Convert to image
    spec_img = ((spec - spec.min()) / (spec.max() - spec.min()) * 255).astype(np.uint8)
    
    # Save as image
    img = Image.fromarray(spec_img)
    img.save(f'dataset/Bird/bird_{i:03d}.png')
    
    if (i + 1) % 20 == 0:
        print(f"  Generated {i + 1} Bird spectrograms")

print("Synthetic dataset generation complete!")
```

---

## Data Quality Checks

### Verify Dataset

```python
from dataset.dataset_loader import DatasetLoader

# Initialize loader
loader = DatasetLoader('./dataset', image_size=(128, 128))

# Load dataset
images, labels, paths = loader.load_dataset()

# Get statistics
info = loader.get_dataset_info()

print("\n" + "=" * 60)
print("DATASET QUALITY CHECK")
print("=" * 60)
print(f"Total images loaded: {len(images)}")
print(f"UAV images: {np.sum(labels == 0)}")
print(f"Bird images: {np.sum(labels == 1)}")
print(f"Image shape: {images[0].shape}")
print(f"Data type: {images.dtype}")
print(f"Min value: {images.min():.4f}")
print(f"Max value: {images.max():.4f}")
print(f"Mean value: {images.mean():.4f}")
print(f"Std deviation: {images.std():.4f}")

# Check for class balance
uav_count = np.sum(labels == 0)
bird_count = np.sum(labels == 1)
balance_ratio = uav_count / bird_count if bird_count > 0 else 0
print(f"\nClass balance ratio (UAV:Bird): {balance_ratio:.2f}")

if uav_count < 100 or bird_count < 100:
    print("\n⚠️  WARNING: Dataset is small (<100 samples per class)")
    print("   Consider generating synthetic data for augmentation")

print("=" * 60 + "\n")
```

### Check Image Integrity

```python
import cv2
from pathlib import Path

def check_image_files():
    """Check all image files for integrity"""
    dataset_dir = Path('./dataset')
    issues = []
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            print(f"Checking {class_dir.name}/ directory...")
            
            for image_path in class_dir.glob('*.png'):
                img = cv2.imread(str(image_path))
                
                if img is None:
                    issues.append(f"Cannot read: {image_path}")
                elif img.shape[0] < 50 or img.shape[1] < 50:
                    issues.append(f"Too small: {image_path} - {img.shape}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Print first 10
            print(f"  - {issue}")
    else:
        print("✓ All images are valid and properly sized")
    
    return len(issues) == 0

check_image_files()
```

---

## Dataset Statistics

### Expected Dataset Characteristics

```
For DIAT-μSAT Dataset:

Typical Statistics:
┌─────────────────────┬──────────┐
│ Parameter           │ Value    │
├─────────────────────┼──────────┤
│ Total Samples       │ 1000+    │
│ UAV Samples         │ ~500     │
│ Bird Samples        │ ~500     │
│ Image Size (orig)   │ 256×256  │
│ Image Size (resized)│ 128×128  │
│ Image Format        │ PNG/JPG  │
│ Color Mode          │ Grayscale│
│ Value Range         │ [0, 255] │
│ Frequency Range     │ 0-500 Hz │
│ Time Range          │ 0-1 sec  │
│ Class Balance       │ ~1:1     │
└─────────────────────┴──────────┘
```

---

## Data Preprocessing Recommendations

### 1. Image Resizing
The system automatically resizes images to 128×128 pixels. This:
- Reduces computational overhead
- Maintains spatial relationships in spectrograms
- Speeds up training (recommended size for academic projects)

### 2. Normalization
Images are normalized to [0, 1] range:
- Min-max normalization: (x - min) / (max - min)
- Improves model convergence
- Standardizes input ranges

### 3. Augmentation
Data augmentation creates variations:
- Rotation (±15°): Simulates different viewing angles
- Flipping: Increases sample diversity
- Noise addition: Improves robustness
- Intensity adjustment: Handles illumination variations

### 4. Denoising
Optional noise reduction:
- Gaussian blur: Smoothing
- Bilateral filter: Edge-preserving smoothing
- Morphological operations: Structure enhancement

---

## Handling Class Imbalance

If your dataset has unequal classes:

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    np.unique(labels),
    labels
)

class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# Use during training
history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    class_weight=class_weight_dict
)
```

---

## Data Augmentation Strategy

```python
from preprocessing.preprocessing import DataAugmentation

augmentor = DataAugmentation()

# Augment training data
X_train_aug, y_train_aug = augmentor.augment_dataset(
    X_train, y_train,
    augment_factor=3  # Creates 3x more training samples
)

print(f"Original training samples: {len(X_train)}")
print(f"Augmented training samples: {len(X_train_aug)}")
```

---

## Splitting Strategy

### Recommended Data Split
```
Total Dataset (1000 samples)
│
├── Training Set (70% = 700)
│   ├── Training (56% = 560)
│   └── Validation (14% = 140)
│
└── Test Set (30% = 300)
    └── For Final Evaluation
```

### Implementation
```python
from sklearn.model_selection import train_test_split

# First split: 70% train, 30% test
X_temp, X_test, y_temp, y_test = train_test_split(
    images, labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# Second split: 80% train, 20% validation (of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.20,
    stratify=y_temp,
    random_state=42
)

print(f"Training: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Testing: {len(X_test)} samples")
```

---

## Sample Size Recommendations

### Minimum Requirements
```
Per Class:
- Minimum: 50 samples (for testing only)
- Recommended: 200+ samples (for good results)
- Optimal: 500+ samples (for robust model)

Total Dataset:
- Minimum: 100 samples
- Recommended: 500-1000 samples
- Optimal: 1000+ samples
```

### Performance vs. Dataset Size
```
Samples     Accuracy    Training Time   Overfitting Risk
─────────────────────────────────────────────────────────
100         70-75%      2-5 min         Very High
200         75-80%      5-10 min        High
500         80-85%      10-20 min       Medium
1000        85-92%      20-50 min       Low
2000        90-95%      50-100 min      Very Low
```

---

## Downloading from ResearchGate

### Step-by-Step Guide

1. **Visit ResearchGate**
   - Go to: https://www.researchgate.net/
   - Create account if needed

2. **Find the Dataset**
   - Search: "DIAT-μSAT"
   - Look for: "DIAT-mSAT Small Aerial Targets' Micro-Doppler Signatures"
   - Author: Usually published by DIAT or related institutions

3. **Request Access**
   - Click "Request" button if needed
   - Author may approve request
   - Or download if publicly available

4. **Download and Extract**
   - Download ZIP file
   - Extract to `./raw_dataset/`
   - Organize according to directory structure above

---

## Creating Your Own Dataset

If you want to create a custom dataset:

```python
# Step 1: Acquire raw radar signals
# (Use radar simulation software or hardware)

# Step 2: Generate spectrograms
from spectrogram.spectrogram import SpectrogramGenerator

generator = SpectrogramGenerator(sampling_rate=1000, n_fft=512)

# Load your raw signal
raw_signal = load_radar_signal('path/to/signal.bin')

# Preprocess
processed = generator.preprocess_signal(raw_signal)

# Generate spectrogram
f, t, spectrogram = generator.compute_stft(processed)

# Save as image
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, spectrogram, shading='gouraud', cmap='viridis')
plt.savefig('spectrogram.png', dpi=100, bbox_inches='tight')

# Step 3: Organize in dataset structure
# Move PNG files to dataset/UAV/ or dataset/Bird/
```

---

## Troubleshooting Dataset Issues

### Issue: "No images found"
```python
# Check directory structure
import os
uav_files = os.listdir('dataset/UAV')
bird_files = os.listdir('dataset/Bird')
print(f"UAV files: {len(uav_files)}")
print(f"Bird files: {len(bird_files)}")
```

### Issue: "Images not loading"
```python
# Verify image files
import cv2
img = cv2.imread('dataset/UAV/image.png', cv2.IMREAD_GRAYSCALE)
print(f"Image shape: {img.shape if img is not None else 'Failed to load'}")
```

### Issue: "Dataset too small"
```python
# Generate synthetic augmentation
from preprocessing.preprocessing import DataAugmentation
augmentor = DataAugmentation()
X_aug, y_aug = augmentor.augment_dataset(images, labels, augment_factor=5)
```

### Issue: "Class imbalance"
```python
# Use class weights during training
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', np.unique(labels), labels)
```

---

## Summary

✓ Download DIAT-μSAT dataset (or use synthetic data)
✓ Organize in `dataset/UAV/` and `dataset/Bird/`
✓ Verify dataset with `check_image_files()` function
✓ Check statistics with `DatasetLoader`
✓ Run system with `UAVBirdClassificationSystem()`

Your dataset is now ready for training!

---

**For questions on dataset preparation, refer to:**
- `dataset/dataset_loader.py` docstrings
- `README.md` Dataset Details section
- `ARCHITECTURE.md` Data Flow Diagram
