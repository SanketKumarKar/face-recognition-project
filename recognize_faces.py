import cv2
import numpy as np
import tensorflow as tf
import os
import time
import dlib
from imutils import face_utils

class FaceRecognizer:
    def __init__(self):
        # Load the trained model if it exists
        self.model_path = 'face_recognition_model.h5'
        self.model_loaded = False
        
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_loaded = True
            print("Model loaded successfully")
        else:
            print(f"Warning: Model file {self.model_path} not found")
            
        # Load face detection tools
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize dlib's face detector and facial landmark predictor if available
        self.use_dlib = False
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        if os.path.exists(predictor_path):
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(predictor_path)
                self.use_dlib = True
                print("Using dlib for enhanced face detection")
            except Exception as e:
                print(f"Error loading dlib components: {e}")
                
        # Names for the recognized individuals
        self.face_names = ["Unknown", "Person 1", "Person 2", "Person 3", "Person 4", "Person 5"]
        
    def align_face(self, image, face):
        """Align the face using facial landmarks if dlib is available"""
        if not self.use_dlib:
            return face
            
        x, y, w, h = face
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Get landmarks for eyes
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # Calculate center of each eye
            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")
            
            # Calculate angle between eyes
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Get center of face for rotation
            center = (x + w // 2, y + h // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1)
            
            # Apply rotation to whole image
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # Extract the rotated face
            aligned_face = rotated[y:y+h, x:x+w]
            return aligned_face
        except Exception as e:
            print(f"Error aligning face: {e}")
            return face
        
    def preprocess_face(self, face):
        """Preprocess face for model input"""
        try:
            face_resized = cv2.resize(face, (224, 224))
            face_normalized = face_resized / 255.0
            return face_normalized
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
            
    def detect_faces(self, frame):
        """Detect faces in a frame using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
        
    def recognize_face(self, face):
        """Recognize a face using the loaded model"""
        if not self.model_loaded:
            return None, 0.0
            
        try:
            preprocessed_face = self.preprocess_face(face)
            if preprocessed_face is None:
                return None, 0.0
                
            # Add batch dimension
            face_batch = np.expand_dims(preprocessed_face, axis=0)
            
            # Get model prediction
            predictions = self.model.predict(face_batch, verbose=0)
            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id])
            
            name = self.face_names[class_id] if class_id < len(self.face_names) else "Unknown"
            
            return name, confidence
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return None, 0.0
            
    def process_frame(self, frame):
        """Process a frame to detect and recognize faces"""
        # Make a copy for drawing
        display_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # List to store recognition results
        results = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Get face region
            face_region = frame[y:y+h, x:x+w]
            
            # Align face if possible
            aligned_face = self.align_face(frame, (x, y, w, h))
            
            # Only perform recognition if model is loaded
            if self.model_loaded:
                name, confidence = self.recognize_face(aligned_face)
                
                if name and confidence > 0.6:
                    label = f"{name}: {confidence:.2f}"
                    results.append({
                        "name": name,
                        "confidence": confidence,
                        "time": time.strftime("%H:%M:%S"),
                        "position": (x, y, w, h)
                    })
                else:
                    label = "Unknown"
                    
                # Add text above face rectangle
                cv2.putText(display_frame, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        return display_frame, results

# Example usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result_frame, results = recognizer.process_frame(frame)
        
        cv2.imshow('Face Recognition', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
