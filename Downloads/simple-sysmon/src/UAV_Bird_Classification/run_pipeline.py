#!/usr/bin/env python3
"""
Execute Complete UAV vs Bird Classification Pipeline
This runs the entire system: load → preprocess → train → evaluate → report
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required = ['tensorflow', 'numpy', 'cv2', 'sklearn', 'pandas', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing packages:", ', '.join(missing))
        print("\n📦 Install with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_dataset():
    """Check if dataset exists"""
    uav_dir = Path('dataset/UAV')
    bird_dir = Path('dataset/Bird')
    
    uav_images = list(uav_dir.glob('*.png')) + list(uav_dir.glob('*.jpg'))
    bird_images = list(bird_dir.glob('*.png')) + list(bird_dir.glob('*.jpg'))
    
    if len(uav_images) == 0 or len(bird_images) == 0:
        print("❌ Dataset not found or empty!")
        print("\n📊 Options:")
        print("   1. Generate synthetic data: python3 generate_data.py")
        print("   2. Add real images to:")
        print(f"      - {uav_dir.absolute()}")
        print(f"      - {bird_dir.absolute()}")
        return False
    
    print(f"✓ Dataset found: {len(uav_images)} UAV + {len(bird_images)} Bird images")
    return True

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("UAV vs BIRD CLASSIFICATION SYSTEM - COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    # Check dependencies
    print("1️⃣  Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("   ✓ All dependencies installed\n")
    
    # Check dataset
    print("2️⃣  Checking dataset...")
    if not check_dataset():
        sys.exit(1)
    print()
    
    # Import main system
    print("3️⃣  Loading system...")
    try:
        from main import UAVBirdClassificationSystem
        print("   ✓ System loaded\n")
    except Exception as e:
        print(f"   ❌ Error loading system: {str(e)}")
        sys.exit(1)
    
    # Create system with optimized config
    print("4️⃣  Initializing system...")
    config = {
        'dataset_path': 'dataset/',
        'image_size': 128,
        'batch_size': 16,
        'epochs': 30,              # Reduced for testing/demo
        'test_split': 0.2,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'augment_data': True,
        'augment_factor': 2,
        'verbose': True
    }
    
    try:
        system = UAVBirdClassificationSystem(config=config)
        print("   ✓ System initialized\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        sys.exit(1)
    
    # Run pipeline
    print("5️⃣  Running complete pipeline...")
    print("   " + "-"*76)
    
    try:
        result = system.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("✅ PIPELINE EXECUTION COMPLETE!")
        print("="*80 + "\n")
        
        # Summary
        print("📊 RESULTS SUMMARY:")
        print("-"*80)
        print(f"✓ Model trained and saved to: model/trained_model.h5")
        print(f"✓ Database saved to: database/predictions.db")
        print(f"✓ Reports generated in: reports/")
        print()
        
        # Files generated
        print("📁 GENERATED FILES:")
        print("-"*80)
        reports_dir = Path('reports')
        if reports_dir.exists():
            for report_file in sorted(reports_dir.glob('*')):
                size_kb = report_file.stat().st_size / 1024
                print(f"   ✓ {report_file.name:<40} ({size_kb:.1f} KB)")
        print()
        
        # Model info
        model_path = Path('model/trained_model.h5')
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            print(f"   ✓ {model_path.name:<40} ({size_mb:.1f} MB)")
        print()
        
        # Next steps
        print("🚀 NEXT STEPS:")
        print("-"*80)
        print("1. View training results:")
        print("   cat reports/summary_report.txt")
        print()
        print("2. Check evaluation metrics:")
        print("   cat reports/evaluation_report.csv")
        print()
        print("3. Make predictions on new images:")
        print("   python3 predict_new.py <image_path>")
        print()
        print("4. Push to GitHub:")
        print("   git add . && git commit -m 'Add trained model' && git push")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
