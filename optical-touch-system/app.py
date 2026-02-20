import eventlet
eventlet.monkey_patch()
import threading
import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from PIL import Image
import io
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'optical_touch_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directories
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'presentations'), exist_ok=True)
os.makedirs('calibration', exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables
calibration_points = []
calibration_step = 0
calibration_data = None
touch_points = []
last_touch_time = {}
touch_threshold = 50  # Minimum distance between touch points
touch_timeout = 2  # Seconds before removing touch point

# Touch detection parameters
background = None
bg_update_rate = 0.05  # Background update rate
touch_sensitivity = 30  # Lower = more sensitive
min_touch_area = 50
max_touch_area = 1000

class TouchDetector:
    def __init__(self):
        self.background = None
        self.bg_alpha = 0.95  # Background averaging factor
        self.min_contour_area = 50
        self.max_contour_area = 1000
        self.threshold_value = 25
        self.kernel = np.ones((5,5), np.uint8)
        
    def update_background(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.background is None:
            self.background = gray.astype("float")
        else:
            cv2.accumulateWeighted(gray, self.background, self.bg_alpha)
    
    def detect_touches(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.background is None:
            return []
        
        # Compute absolute difference between current frame and background
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        
        # Apply threshold
        thresh = cv2.threshold(frame_delta, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to remove noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        touch_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    touch_points.append((cx, cy))
        
        return touch_points

detector = TouchDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/projector')
def projector():
    return render_template('projector.html')

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global calibration_step, calibration_points, calibration_data
    
    data = request.json
    step = data.get('step')
    
    if step == 'start':
        calibration_step = 0
        calibration_points = []
        return jsonify({'status': 'started', 'step': 0})
    
    elif step == 'point':
        point = data.get('point')
        if point:
            calibration_points.append(point)
            calibration_step += 1
            
            if calibration_step >= 4:
                # Calculate transformation matrix
                screen_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
                camera_points = np.array(calibration_points, dtype=np.float32)
                
                # Calculate perspective transform
                matrix = cv2.getPerspectiveTransform(camera_points, screen_points)
                
                calibration_data = {
                    'matrix': matrix.tolist(),
                    'calibration_points': calibration_points,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save calibration data
                with open('calibration/calibration_data.json', 'w') as f:
                    json.dump(calibration_data, f)
                
                return jsonify({'status': 'calibrated', 'matrix': matrix.tolist()})
            
            return jsonify({'status': 'next', 'step': calibration_step})
    
    return jsonify({'status': 'error', 'message': 'Invalid calibration step'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_type = request.form.get('type', 'pdf')
    
    if file_type == 'pdf':
        filename = f"document_{int(time.time())}.pdf"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/uploads/pdfs/{filename}',
            'type': 'pdf'
        })
    
    elif file_type == 'presentation':
        # For simplicity, we'll handle images as presentation slides
        filename = f"slide_{int(time.time())}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'presentations', filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/uploads/presentations/{filename}',
            'type': 'image'
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Update background model
        detector.update_background(frame)
        
        # Detect touches
        touches = detector.detect_touches(frame)
        
        # Transform coordinates if calibrated
        screen_touches = []
        if calibration_data:
            matrix = np.array(calibration_data['matrix'])
            for touch in touches:
                pt = np.array([[[touch[0], touch[1]]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, matrix)
                x, y = transformed[0][0]
                screen_touches.append({'x': float(x), 'y': float(y)})
        
        # Send touches to all projector clients
        emit('touch_update', {'touches': screen_touches}, broadcast=True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")

@socketio.on('draw_action')
def handle_draw_action(data):
    # Broadcast drawing actions to all projector clients
    emit('draw_update', data, broadcast=True)

@socketio.on('clear_canvas')
def handle_clear_canvas():
    emit('canvas_cleared', broadcast=True)

@socketio.on('load_document')
def handle_load_document(data):
    emit('document_loaded', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)