import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
import time
from collections import defaultdict
from typing import Dict, List, Tuple

class ShopliftingDetector:
    def __init__(self, video_source=0):
        # Initialize YOLO model
        self.model = YOLO("yolov8n.pt")
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=50)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        
        # Detection parameters
        self.loitering_threshold = 10  # seconds
        self.concealment_distance = 50  # pixels
        
        # Tracking dictionaries
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.track_start_times: Dict[int, float] = {}
        self.suspicious_tracks: Dict[int, str] = {}

    def calculate_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def detect_loitering(self, track_id: int, current_time: float) -> bool:
        if track_id in self.track_start_times:
            duration = current_time - self.track_start_times[track_id]
            if duration > self.loitering_threshold:
                return True
        return False

    def detect_concealment(self, person_bbox, object_bbox) -> bool:
        person_center = self.calculate_bbox_center(person_bbox)
        object_center = self.calculate_bbox_center(object_bbox)
        
        distance = np.sqrt(
            (person_center[0] - object_center[0])**2 + 
            (person_center[1] - object_center[1])**2
        )
        return distance < self.concealment_distance

    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > 0.3:  # Confidence threshold
                detections.append(([x1, y1, x2, y2], score, int(class_id)))

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        current_time = time.time()

        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # top left bottom right format

            # Initialize tracking for new objects
            if track_id not in self.track_start_times:
                self.track_start_times[track_id] = current_time

            # Update track history
            center = self.calculate_bbox_center(bbox)
            self.track_history[track_id].append(center)

            # Check for suspicious behavior
            if self.detect_loitering(track_id, current_time):
                self.suspicious_tracks[track_id] = "Loitering"

            # Draw bounding box
            color = (0, 255, 0)  # Default color (green)
            if track_id in self.suspicious_tracks:
                color = (0, 0, 255)  # Red for suspicious tracks
                
            cv2.rectangle(frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Add label
            label = f"ID: {track_id}"
            if track_id in self.suspicious_tracks:
                label += f" ({self.suspicious_tracks[track_id]})"
            cv2.putText(frame, label, 
                       (int(bbox[0]), int(bbox[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)

            # Draw motion trail
            if len(self.track_history[track_id]) > 2:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display frame
            cv2.imshow('Shoplifting Detection', processed_frame)

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize with default camera (0) or video file path
    detector = ShopliftingDetector(0)
    detector.run()
