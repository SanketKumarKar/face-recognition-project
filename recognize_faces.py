import cv2
import tensorflow as tf
import numpy as np
import dlib

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

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect facial landmarks
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        landmarks = predictor(gray, rect)

        # Align face
        aligned_face = frame[y:y+h, x:x+w]

        # Preprocess face
        preprocessed_face = preprocess_face(aligned_face)
        preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

        # Recognize face
        predictions = model.predict(preprocessed_face)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]

        # Display recognition results
        label = f"ID: {class_id}, Confidence: {confidence:.2f}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
