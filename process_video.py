import cv2
import numpy as np
import os
import time
import json
from recognize_faces import FaceRecognizer
from datetime import datetime

class VideoProcessor:
    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.output_folder = "static/processed_videos"
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
    def process_video(self, video_path):
        """Process a video file for face recognition"""
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{timestamp}_processed.mp4"
        json_filename = f"{timestamp}_results.json"
        
        # Define output video writer
        output_path = os.path.join(self.output_folder, video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize variables to store results
        all_results = []
        unique_faces = set()
        frame_count = 0
        processed_count = 0
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 5th frame to improve performance
            if frame_count % 5 == 0:
                processed_count += 1
                
                # Process the frame
                result_frame, results = self.recognizer.process_frame(frame)
                
                # Write frame to output video
                out.write(result_frame)
                
                # Record results
                if results:
                    for result in results:
                        # Add frame number to result
                        result_with_frame = result.copy()
                        result_with_frame["frame"] = frame_count
                        all_results.append(result_with_frame)
                        
                        # Add to unique faces
                        if result["name"] != "Unknown":
                            unique_faces.add(result["name"])
            else:
                # Write original frame without processing
                out.write(frame)
                
        # Release resources
        cap.release()
        out.release()
        
        # Create summary of results
        summary = {
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "unique_faces_detected": list(unique_faces),
            "total_detections": len(all_results),
            "detailed_results": all_results
        }
        
        # Save results to JSON file
        json_path = os.path.join(self.output_folder, json_filename)
        with open(json_path, 'w') as f:
            json.dump(summary, f)
        
        return {
            "success": True,
            "video_path": output_path,
            "results_path": json_path,
            "summary": {
                "total_frames": frame_count,
                "processed_frames": processed_count,
                "unique_faces": list(unique_faces),
                "total_detections": len(all_results)
            }
        }

if __name__ == "__main__":
    processor = VideoProcessor()
    result = processor.process_video("sample_video.mp4")
    print(result)