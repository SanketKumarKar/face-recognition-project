import cv2
import numpy as np

def preprocess_face(face):
    # Resize the face to 224x224 pixels
    face_resized = cv2.resize(face, (224, 224))

    # Normalize the face
    face_normalized = face_resized / 255.0

    return face_normalized
