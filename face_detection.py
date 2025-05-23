import cv2
import dlib

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Dlib's facial landmark detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

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

        # Draw facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('Face Detection and Alignment', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
