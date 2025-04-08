import os
import sys
import json
import base64
import torch
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import io
from utils import load_image, preprocess_image, postprocess_image

# Add YOLOv5 to path
YOLOV5_PATH = Path('yolov5')
if not YOLOV5_PATH.exists():
    print("Error: YOLOv5 directory not found")
    sys.exit(1)

sys.path.append(str(YOLOV5_PATH))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.datasets import letterbox

app = Flask(__name__)

# Initialize model
device = select_device('')
model = attempt_load('model.pt', map_location=device)
model.eval()
stride = int(model.stride.max())
imgsz = 640

def preprocess_image(img):
    # Resize and pad image
    img = letterbox(img, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def detect_objects(image):
    # Preprocess image
    img = preprocess_image(image)
    
    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45)
    
    # Process detections
    results = []
    for i, det in enumerate(pred[0]):
        if det is not None and len(det):
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            results.append({
                'class': int(cls),
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return results

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        
        # Process image
        results = detect_objects(Image.open(io.BytesIO(img_bytes)))
        
        return jsonify({
            'success': True,
            'detections': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def lambda_handler(event, context):
    try:
        # Parse the request body
        body = json.loads(event['body'])
        if 'image' not in body:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
        
        # Load model if not already loaded
        load_model()
        
        # Process the image
        processed_image = process_image(body['image'])
        
        # Return the processed image
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Image processed successfully',
                'image': processed_image
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def load_model():
    global model
    if model is None:
        model_path = Path('/tmp/model.pt')
        if not model_path.exists():
            raise FileNotFoundError("Model file not found in /tmp/model.pt")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
        model.eval()

def process_image(image_data):
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(image)
    
    # Draw circles around detected objects
    for det in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
        x1, y1, x2, y2 = map(int, det[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2)
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
    
    # Encode the processed image
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 