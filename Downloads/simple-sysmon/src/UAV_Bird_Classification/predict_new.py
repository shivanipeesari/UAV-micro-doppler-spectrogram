#!/usr/bin/env python3
"""
Make predictions on new images using the trained model
Usage: python3 predict_new.py <image_path>
"""

import sys
import os
from pathlib import Path

def predict_single_image(image_path):
    """Predict class and confidence for a single image"""
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Check model exists
    model_path = Path('model/trained_model.h5')
    if not model_path.exists():
        print(f"❌ Error: Trained model not found: {model_path}")
        print("\n💡 Please run: python3 run_pipeline.py")
        sys.exit(1)
    
    print(f"\n🔮 Making prediction...")
    print(f"   Image: {image_path.name}")
    
    try:
        from database.predict import ImagePredictor
        
        # Load predictor
        predictor = ImagePredictor(model_path=str(model_path))
        
        # Make prediction
        result = predictor.predict_single(str(image_path))
        
        # Display results
        print("\n📊 PREDICTION RESULTS:")
        print("-" * 60)
        print(f"  Predicted Class:    {result['class']}")
        print(f"  Confidence:         {result['confidence']:.2%}")
        print(f"  UAV Probability:    {result['probabilities'][0]:.2%}")
        print(f"  Bird Probability:   {result['probabilities'][1]:.2%}")
        print("-" * 60)
        
        # Decision threshold
        if result['meets_threshold']:
            print(f"✅ Prediction meets confidence threshold")
        else:
            print(f"⚠️  Low confidence prediction")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def predict_batch(directory):
    """Predict on all images in a directory"""
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Error: Directory not found: {directory}")
        sys.exit(1)
    
    # Get all images
    image_files = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg'))
    
    if not image_files:
        print(f"❌ No images found in: {directory}")
        sys.exit(1)
    
    print(f"\n🔮 Making batch predictions...")
    print(f"   Directory: {directory}")
    print(f"   Images: {len(image_files)}")
    
    # Check model
    model_path = Path('model/trained_model.h5')
    if not model_path.exists():
        print(f"❌ Error: Trained model not found")
        print("💡 Please run: python3 run_pipeline.py")
        sys.exit(1)
    
    try:
        from database.predict import ImagePredictor
        
        predictor = ImagePredictor(model_path=str(model_path))
        
        # Predict all
        results = predictor.predict_batch([str(f) for f in image_files])
        
        # Display results
        print("\n📊 BATCH PREDICTION RESULTS:")
        print("-" * 80)
        print(f"{'Image':<30} {'Prediction':<15} {'Confidence':<15}")
        print("-" * 80)
        
        uav_count = 0
        bird_count = 0
        total_confidence = 0
        
        for result in results:
            image_name = Path(result['image_path']).name
            pred_class = result['class']
            confidence = result['confidence']
            
            print(f"{image_name:<30} {pred_class:<15} {confidence:.2%}")
            
            if pred_class == 'UAV':
                uav_count += 1
            else:
                bird_count += 1
            
            total_confidence += confidence
        
        # Summary
        print("-" * 80)
        print(f"\n✅ SUMMARY:")
        print(f"   UAV predictions:    {uav_count} ({uav_count/len(results):.1%})")
        print(f"   Bird predictions:   {bird_count} ({bird_count/len(results):.1%})")
        print(f"   Avg confidence:     {total_confidence/len(results):.2%}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("\n📋 USAGE:")
        print("-" * 60)
        print("Single image prediction:")
        print("   python3 predict_new.py <image_path>")
        print()
        print("Batch prediction:")
        print("   python3 predict_new.py --batch <directory>")
        print()
        print("Examples:")
        print("   python3 predict_new.py dataset/UAV/uav_001.png")
        print("   python3 predict_new.py --batch dataset/UAV/")
        print()
        return
    
    # Check for batch flag
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("❌ Error: Please provide directory path")
            print("   Usage: python3 predict_new.py --batch <directory>")
            sys.exit(1)
        predict_batch(sys.argv[2])
    else:
        # Single image
        predict_single_image(sys.argv[1])

if __name__ == '__main__':
    main()
