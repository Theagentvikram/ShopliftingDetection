from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from shoplifting_detector import ShopliftingDetector
import cv2
import numpy as np
from datetime import datetime
import json
from flask_sock import Sock
import base64

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Initialize paths
UPLOAD_FOLDER = 'uploads'
TRAINING_FOLDER = 'training_data'
MODEL_FOLDER = 'models'

for folder in [UPLOAD_FOLDER, TRAINING_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

detector = ShopliftingDetector()

@sock.route('/ws/detect')
def ws_detect(ws):
    while True:
        # Receive frame as base64 string
        frame_data = ws.receive()
        
        # Convert base64 to numpy array
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        processed_frame, results = detector.process_frame(frame)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Send results back to client
        ws.send(json.dumps({
            'frame': f'data:image/jpeg;base64,{processed_frame_data}',
            'results': results
        }))

@app.route('/api/detect', methods=['POST'])
def detect_shoplifting():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, f'video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    video_file.save(video_path)
    
    # Process video and get results
    output_path = os.path.join(UPLOAD_FOLDER, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    results = detector.process_video(video_path, output_path)
    
    # Add output video path to results
    results['output_video'] = output_path
    
    return jsonify(results)

@app.route('/api/train', methods=['POST'])
def train_model():
    if 'video' not in request.files or 'annotations' not in request.files:
        return jsonify({'error': 'Both video and annotations are required'}), 400
    
    video_file = request.files['video']
    annotations_file = request.files['annotations']
    
    # Save files
    video_path = os.path.join(TRAINING_FOLDER, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    annotations_path = os.path.join(TRAINING_FOLDER, f'annotations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    video_file.save(video_path)
    annotations_file.save(annotations_path)
    
    # Train model
    training_results = detector.train(video_path, annotations_path)
    
    return jsonify(training_results)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Get detection statistics
    stats = {
        'total_detections': detector.total_detections,
        'suspicious_activities': detector.suspicious_activities,
        'model_version': detector.model_version
    }
    return jsonify(stats)

@app.route('/api/video/<path:filename>')
def serve_video(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
