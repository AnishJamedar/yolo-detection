import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the given path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
        
    return img

def preprocess_image(img: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess image for YOLO model.
    
    Args:
        img: Input image as numpy array
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image
    """
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def postprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Postprocess image after model inference.
    
    Args:
        img: Model output image
        
    Returns:
        Postprocessed image ready for visualization
    """
    # Remove batch dimension
    img = np.squeeze(img)
    
    # Transpose back to HWC format
    img = np.transpose(img, (1, 2, 0))
    
    # Scale back to [0, 255]
    img = (img * 255).astype(np.uint8)
    
    # Convert back to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img 