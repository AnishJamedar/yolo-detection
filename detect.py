import os
import sys
import torch
from pathlib import Path

# Add YOLOv5 to path
YOLOV5_PATH = Path('yolov5')
if not YOLOV5_PATH.exists():
    print("Error: YOLOv5 directory not found. Please make sure you're in the correct directory.")
    sys.exit(1)

# Add YOLOv5 to Python path
sys.path.append(str(YOLOV5_PATH))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import cv2
import numpy as np

def detect_objects(source, weights='runs/train/exp/weights/best.pt', img_size=640, conf_thres=0.25, iou_thres=0.45):
    """
    Perform object detection on images using trained YOLOv5 model
    
    Args:
        source (str): Path to image or directory of images
        weights (str): Path to model weights
        img_size (int): Input image size
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold for NMS
    """
    # Initialize
    device = select_device('')
    
    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = img_size
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # Process image
    if os.path.isdir(source):
        files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        files = [source]
        
    for path in files:
        # Load image
        img0 = cv2.imread(path)
        if img0 is None:
            print(f"Error loading image: {path}")
            continue
            
        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]
        
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        pred = model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process detections
        for i, det in enumerate(pred):
            im0 = img0.copy()
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print(f"{n} {names[int(c)]}{'s' * (n > 1)}")
                
                # Draw boxes
                annotator = Annotator(im0, line_width=3)
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.result()
            
            # Save results
            output_path = str(Path('output') / Path(path).name)
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(output_path, im0)
            print(f"Results saved to {output_path}")

def main():
    # Example usage
    source = 'test_images'  # Directory containing test images
    if not os.path.exists(source):
        os.makedirs(source)
        print(f"Created {source} directory. Please add some test images there.")
        return
        
    detect_objects(source)

if __name__ == '__main__':
    main() 