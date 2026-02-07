#!/usr/bin/env python3
"""
Generate synthetic test data for quick testing
Run this before running the main pipeline
"""

import os
import numpy as np
import cv2
from pathlib import Path

def create_synthetic_dataset(uav_count=30, bird_count=30):
    """Create synthetic spectrogram images for testing"""
    
    # Create directories
    Path('dataset/UAV').mkdir(parents=True, exist_ok=True)
    Path('dataset/Bird').mkdir(parents=True, exist_ok=True)
    
    print("🔄 Generating synthetic test spectrograms...")
    
    # Generate UAV spectrograms (multi-harmonic, clean)
    print(f"  Generating {uav_count} UAV spectrograms...", end=" ", flush=True)
    for i in range(uav_count):
        # Create UAV-like spectrogram (harmonic pattern)
        img = np.zeros((128, 128), dtype=np.uint8)
        
        # Add horizontal lines (harmonic series)
        base_freq = 30 + np.random.randint(-5, 5)
        for harmonic in range(1, 4):
            freq_pos = int(base_freq * harmonic)
            if freq_pos < 128:
                # Draw thick band
                img[max(0, freq_pos-2):min(128, freq_pos+2), :] = np.random.randint(200, 256)
        
        # Add some noise
        noise = np.random.normal(0, 10, img.shape)
        img = cv2.GaussianBlur(img.astype(np.float32) + noise, (3, 3), 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f'dataset/UAV/uav_{i:03d}.png', img)
    print("✓")
    
    # Generate Bird spectrograms (modulated, complex)
    print(f"  Generating {bird_count} Bird spectrograms...", end=" ", flush=True)
    for i in range(bird_count):
        # Create Bird-like spectrogram (wing flapping modulation)
        img = np.zeros((128, 128), dtype=np.uint8)
        
        # Add modulated pattern (wing beat)
        base_freq = 45 + np.random.randint(-5, 5)
        for t in range(128):
            # Frequency modulation (wing beat)
            freq = base_freq + 15 * np.sin(2 * np.pi * t / 128)
            freq_pos = int(freq)
            if 0 <= freq_pos < 128:
                img[freq_pos, t] = np.random.randint(180, 256)
        
        # Blur and add noise
        img = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)
        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f'dataset/Bird/bird_{i:03d}.png', img)
    print("✓")
    
    # Verify
    uav_images = len(list(Path('dataset/UAV').glob('*.png')))
    bird_images = len(list(Path('dataset/Bird').glob('*.png')))
    
    print("\n✅ Dataset ready!")
    print(f"  UAV images: {uav_images}")
    print(f"  Bird images: {bird_images}")
    print(f"  Total: {uav_images + bird_images}")
    
    return uav_images, bird_images

if __name__ == '__main__':
    try:
        create_synthetic_dataset(uav_count=30, bird_count=30)
        print("\n🚀 Run next: python3 run_pipeline.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
