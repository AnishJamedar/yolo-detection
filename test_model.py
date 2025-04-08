#!/usr/bin/env python3
"""
Test script to verify YOLO model loading works correctly.
Run this script before deploying to ensure compatibility.
"""

import sys
import os
from model_loader import load_yolo_model

def main():
    print("Testing YOLO model loading...")
    try:
        # Try to load the model
        model = load_yolo_model('yolov8n.pt')
        print("✅ Model loaded successfully!")
        
        # Test a simple inference
        print("\nTesting inference with a blank image...")
        import numpy as np
        import cv2
        
        # Create a blank image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(img)
        print("✅ Inference successful!")
        print(f"Detected {len(results[0].boxes)} objects in the blank image")
        
        print("\nAll tests passed! Your model is ready for deployment.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nModel loading failed. Please check the error message above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 