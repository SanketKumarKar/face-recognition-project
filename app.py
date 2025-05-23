from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import tensorflow as tf
import numpy as np
import dlib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('face_recognition_model.h5')

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Dlib's facial landmark detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Function to preprocess face
def preprocess_face(face):
    face_resized = cv2.resize(face, (224, 224))
    face_normalized = face_resized / 255.0
    return face_normalized

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                landmarks = predictor(gray, rect)
                aligned_face = frame[y:y+h, x:x+w]
                preprocessed_face = preprocess_face(aligned_face)
                preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
                predictions = model.predict(preprocessed_face)
                class_id = np.argmax(predictions)
                confidence = predictions[0][class_id]
                label = f"ID: {class_id}, Confidence: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
