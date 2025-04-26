import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load pre-trained facial expression recognition model
model = load_model('my_model_last.h5')


# Define labels for the facial expressions
class_labels = ['Not Stress', 'Stress']

# Function to detect and classify facial expressions
def detect_emotion(face_img):
    # Preprocess the image for the model
    face_img = cv2.resize(face_img, (48, 48))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0

    # Predict the emotion
    predictions = model.predict(face_img)
    emotion_label = class_labels[np.argmax(predictions)]

    return emotion_label

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define colors for text and rectangle
text_color = (255, 255, 255)  # White
rectangle_color = (255, 0, 0)  # Blue

# Main loop to capture and process video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = gray[y:y+h, x:x+w]

        # Classify the facial expression
        emotion_label = detect_emotion(face_img)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)

        # Display the detected expression above the rectangle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # Display the resulting frame
    cv2.imshow('Stress Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
