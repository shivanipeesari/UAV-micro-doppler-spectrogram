#!/usr/bin/env python3
"""
Quick Start Script - Automated Setup and Execution
Cross-platform compatible (Windows, macOS, Linux)
Run this once and it does everything for you!
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(args, description):
    """
    Run a command in a cross-platform way.
    Uses subprocess without shell=True for better portability.
    
    Args:
        args: List of command arguments
        description: Human-readable description
    """
    print(f"\n📋 {description}...")
    try:
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ Success")
            return True
        else:
            error_msg = result.stderr[:100] if result.stderr else "Unknown error"
            print(f"   ❌ Failed: {error_msg}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    """Main quick start flow - cross-platform compatible"""
    
    print("\n" + "="*80)
    print("🚀 UAV vs BIRD CLASSIFICATION - QUICK START")
    print("="*80)
    print("\nThis script will:")
    print("  1. Check Python version")
    print("  2. Install dependencies")
    print("  3. Generate test dataset")
    print("  4. Run complete pipeline")
    print("  5. Display results")
    print(f"\n💻 Platform: {sys.platform}")
    print(f"🐍 Python: {sys.executable}\n")
    
    # Step 1: Check Python
    print("\n1️⃣  Checking Python version...")
    version = sys.version.split()[0]
    print(f"   ✓ Python {version}")
    
    if sys.version_info < (3, 8):
        print(f"   ❌ Python 3.8+ required")
        sys.exit(1)
    
    # Step 2: Install dependencies
    print("\n2️⃣  Installing dependencies...")
    if not run_command([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], 
                       "Installing packages"):
        print("   💡 Try: python -m pip install -r requirements.txt")
    
    # Step 3: Create dataset
    print("\n3️⃣  Preparing dataset...")
    if not run_command([sys.executable, "generate_data.py"], "Generating synthetic data"):
        print("   ❌ Could not generate data")
        sys.exit(1)
    
    # Step 4: Run pipeline
    print("\n4️⃣  Running pipeline...")
    print("   " + "-"*76)
    
    if not run_command([sys.executable, "run_pipeline.py"], "Executing full pipeline"):
        print("   ❌ Pipeline execution failed")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ ALL STEPS COMPLETE!")
    print("="*80)
    print("\n📊 Results are in:")
    print("   - reports/summary_report.txt")
    print("   - reports/training_report.csv")
    print("   - reports/evaluation_report.csv")
    print("   - model/trained_model.h5")
    print("\n🚀 Next steps:")
    print("   1. View results: cat reports/summary_report.txt")
    print("   2. Make predictions: python predict_new.py dataset/UAV/uav_001.png")
    print("   3. Push to GitHub: git add . && git commit -m 'Add results' && git push")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
