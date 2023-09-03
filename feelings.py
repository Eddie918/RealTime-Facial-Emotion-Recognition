import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Try to load the model using an absolute path
try:
    model = load_model("/Users/s.a.u.1.ariasss/Desktop/recognitio /simplified_emotion_model.h5")
except OSError as e:
    print("An error occurred:", e)

# Load the pre-trained model for facial emotion recognition
model = load_model("simplified_emotion_model.h5")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Flip the frame horizontally (Mirror the image)
    frame = cv2.flip(frame, 1)  # This line flips the image horizontally

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        face = face / 255.0

        # Make prediction
        predictions = model.predict(face)[0]
        max_index = np.argmax(predictions)
        emotion_detection = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_prediction = emotion_detection[max_index]

        # Display result
        cv2.putText(frame, emotion_prediction, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Stop the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
