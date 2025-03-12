from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from shoplifting_detector import ShopliftingDetector
import cv2
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Initialize paths
UPLOAD_FOLDER = 'uploads'
TRAINING_FOLDER = 'training_data'
MODEL_FOLDER = 'models'

for folder in [UPLOAD_FOLDER, TRAINING_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

detector = ShopliftingDetector()

@app.route('/api/detect', methods=['POST'])
def detect_shoplifting():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, f'video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    video_file.save(video_path)
    
    # Process video and get results
    results = detector.process_video(video_path)
    
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
