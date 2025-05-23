from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, flash, session
import cv2
import torch  # Using PyTorch instead of TensorFlow
import numpy as np
import dlib
import time
import os
import uuid
from datetime import datetime
from recognize_faces import FaceRecognizer
from process_video import VideoProcessor
import threading
import queue

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit for uploads

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/processed_videos', exist_ok=True)
os.makedirs('static/snapshots', exist_ok=True)

# Initialize the face recognizer
recognizer = FaceRecognizer()
video_processor = VideoProcessor()

# Store recognition results for display
latest_results = []
processing_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html', model_loaded=recognizer.model_loaded)

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    frame_count = 0
    global latest_results
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every other frame to improve performance
        if frame_count % 2 == 0:
            # Process the frame
            display_frame, results = recognizer.process_frame(frame)
            
            # Update latest results if we have new ones
            if results:
                latest_results = results
                
        else:
            display_frame = frame
                
        # Convert the image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results_feed():
    def generate():
        global latest_results
        last_sent = []
        
        while True:
            # Only send updates when results change
            if latest_results != last_sent:
                last_sent = latest_results.copy()
                yield f"data: {jsonify(latest_results).get_data(as_text=True)}\n\n"
            
            time.sleep(0.5)  # Check for updates every half second
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
        
    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    # Generate unique filename
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file
    file.save(filepath)
    
    # Process video in a background thread to avoid blocking
    def process_in_background(filepath):
        result = video_processor.process_video(filepath)
        # Store result in session for later retrieval
        processing_queue.put(result)
    
    # Start background thread
    threading.Thread(target=process_in_background, args=(filepath,)).start()
    
    # Set a flag to show processing message
    session['processing_video'] = True
    
    return redirect(url_for('process_status'))

@app.route('/process_status')
def process_status():
    # Check if processing is complete
    try:
        result = processing_queue.get_nowait()
        session['processing_video'] = False
        return render_template('results.html', result=result)
    except queue.Empty:
        # Still processing
        return render_template('processing.html')

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    """Save the current camera frame as a snapshot"""
    try:
        # Access the camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"success": False, "error": "Could not capture frame"})
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join('static/snapshots', filename)
        
        # Process the frame for face detection
        display_frame, results = recognizer.process_frame(frame)
        
        # Save the processed image
        cv2.imwrite(filepath, display_frame)
        
        return jsonify({
            "success": True, 
            "filename": filename, 
            "path": filepath,
            "faces_detected": len(results) if results else 0
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/model_status')
def model_status():
    """Return the status of the face recognition model"""
    return jsonify({
        "model_loaded": recognizer.model_loaded,
        "use_dlib": recognizer.use_dlib
    })

if __name__ == '__main__':
    app.run(debug=True)
