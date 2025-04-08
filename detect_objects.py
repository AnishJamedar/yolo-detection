import cv2
import numpy as np
from ultralytics import YOLO
import os

def draw_circles(image, detections):
    """Draw circles around detected objects."""
    img = image.copy()
    for detection in detections:
        boxes = detection.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = int(max(x2 - x1, y2 - y1) / 2)
            cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
    return img

def process_image(model, image_path, output_path=None):
    """Process a single image and draw circles around detected objects."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to RGB for YOLO
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(img_rgb)
    
    # Draw circles
    img_with_circles = draw_circles(img_rgb, results)
    
    # Convert back to BGR for saving
    img_with_circles = cv2.cvtColor(img_with_circles, cv2.COLOR_RGB2BGR)
    
    # Save or return the result
    if output_path:
        cv2.imwrite(output_path, img_with_circles)
        print(f"Saved result to {output_path}")
    
    return img_with_circles

def main():
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Process all images in the current directory
    for filename in os.listdir('.'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = filename
            output_path = os.path.join('output', f'circles_{filename}')
            
            print(f"Processing {input_path}...")
            process_image(model, input_path, output_path)

if __name__ == "__main__":
    main() 