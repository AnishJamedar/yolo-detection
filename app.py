from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from init_app import init_app

# Initialize the application
init_app()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize YOLO model
model = YOLO('yolov8n.pt')

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Run YOLO detection
    results = model(img)
    
    # Store detection information
    detections = []
    
    # Draw circles around detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2)
            cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
            
            # Get class name and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    
    # Save the processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        output_path, detections = process_image(filepath)
        output_filename = os.path.basename(output_path)
        
        # Return the URL for the processed image and detection information
        return jsonify({
            'message': 'Image processed successfully',
            'image_url': f'/uploads/{output_filename}',
            'detections': detections
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Try port 5000 first
        app.run(debug=True, host='0.0.0.0', port=5000)
    except OSError:
        try:
            # If port 5000 is in use, try port 5001
            print("Port 5000 is in use, trying port 5001...")
            app.run(debug=True, host='0.0.0.0', port=5001)
        except OSError:
            # If both ports are in use, try port 5002
            print("Port 5001 is in use, trying port 5002...")
            app.run(debug=True, host='0.0.0.0', port=5002) 