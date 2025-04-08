import os
import sys
import subprocess
from pathlib import Path

def train_yolo(epochs=100, batch_size=16, img_size=640):
    """
    Train YOLOv5 model on COCO128 dataset using command-line interface
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
    """
    # Check if YOLOv5 directory exists
    yolov5_path = Path('yolov5')
    if not yolov5_path.exists():
        print("Error: YOLOv5 directory not found. Please make sure you're in the correct directory.")
        sys.exit(1)
    
    # Training configuration
    data_yaml = yolov5_path / 'data/coco128.yaml'
    weights = 'yolov5s.pt'  # Use small YOLOv5 model
    project = 'runs/train'
    name = 'exp'
    
    # Start training
    print(f"Starting YOLOv5 training with {epochs} epochs...")
    print(f"Using dataset: {data_yaml}")
    print(f"Model weights: {weights}")
    
    # Build command
    cmd = [
        'python',
        str(yolov5_path / 'train.py'),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', str(data_yaml),
        '--weights', weights,
        '--project', project,
        '--name', name,
        '--exist-ok'
    ]
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed!")
        print(f"Results saved in {project}/{name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def main():
    # Start training
    train_yolo()

if __name__ == '__main__':
    main() 