import requests
import os
import json

# Define the API endpoint
url = "http://127.0.0.1:8000/analyze-video"

# Path to the test video file
video_path = "videoplayback.mp4"

if not os.path.exists(video_path):
    print(f"Error: Video file {video_path} not found!")
    exit(1)

print(f"Testing video processing with file: {video_path}")

# Prepare the file for upload
files = {
    'file': (os.path.basename(video_path), open(video_path, 'rb'), 'video/mp4')
}

try:
    # Send the request
    print("Sending request to backend API...")
    response = requests.post(url, files=files)
    
    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        result = response.json()
        
        print(f"\nDetection Summary:")
        print(f"Response type: {type(result).__name__}")
        
        # Handle different response formats
        if isinstance(result, dict):
            # Dictionary response format
            frame_count = result.get('frame_count', 'N/A')
            fps = result.get('fps', 'N/A')
            
            # Safely handle detections which might be a list
            detections = result.get('detections', [])
            detections_count = len(detections) if isinstance(detections, list) else 0
            
            # Safely handle suspicious activities which might be a list
            suspicious_activities = result.get('suspicious_activities', [])
            suspicious_count = len(suspicious_activities) if isinstance(suspicious_activities, list) else 0
            
            print(f"Total frames processed: {frame_count}")
            print(f"FPS: {fps}")
            print(f"Total detections: {detections_count}")
            print(f"Suspicious activities: {suspicious_count}")
            
            # Print details of suspicious activities
            if suspicious_activities and isinstance(suspicious_activities, list) and suspicious_count > 0:
                print("\nSuspicious Activities:")
                for i, activity in enumerate(suspicious_activities):
                    if isinstance(activity, dict):
                        frame = activity.get('frame', 'N/A')
                        timestamp = activity.get('timestamp', 'N/A')
                        activity_type = activity.get('type', 'N/A')
                        details = activity.get('details', 'N/A')
                        
                        print(f"  {i+1}. Frame {frame}, Time: {timestamp}s")
                        print(f"     Type: {activity_type}")
                        print(f"     Details: {details}")
                    else:
                        print(f"  {i+1}. Invalid activity format: {activity}")
        
        elif isinstance(result, list):
            # List response format
            print(f"Received a list response with {len(result)} items")
            
            # Print the first few items to understand the structure
            for i, item in enumerate(result[:5]):
                print(f"Item {i+1}: {item}")
                if i >= 4 and len(result) > 5:
                    print(f"... and {len(result) - 5} more items")
                    break
        
        else:
            # Unknown response format
            print(f"Unexpected response format: {type(result).__name__}")
            print(f"Response content: {result}")
        
        # Save the full response to a file for further analysis
        with open('detection_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\nFull results saved to detection_results.json")
        
    else:
        print(f"Request failed with status code {response.status_code}")
        print(f"Error message: {response.text}")
        
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Close the file
    files['file'][1].close()
