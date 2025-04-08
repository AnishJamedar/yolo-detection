import os
import torch
import sys
from pathlib import Path

def load_yolo_model(model_path='yolov8n.pt'):
    """
    Load a YOLO model with compatibility for cloud environments like Render.com.
    Uses a simpler approach that doesn't rely on PyTorch's serialization features.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded YOLO model
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        # The model will be downloaded automatically by YOLO
    
    # Import YOLO
    from ultralytics import YOLO
    
    # Load the model with a simpler approach
    try:
        # First try: direct loading
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            # Second try: with explicit weights_only=False
            model = YOLO(model_path, task='detect')
            print("Successfully loaded model with explicit task specification")
            return model
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            try:
                # Third try: with minimal configuration
                model = YOLO('yolov8n.pt')  # Let it download fresh
                print("Successfully loaded model with fresh download")
                return model
            except Exception as e3:
                print(f"All attempts failed. Last error: {e3}")
                raise

if __name__ == "__main__":
    # Test the model loader
    try:
        model = load_yolo_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1) 