import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/mobilenet_finetuned_week4.h5')
labels = ['none', 'paper', 'rock', 'scissors']

# Webcam
cap = cv2.VideoCapture(0)

print("Rock–Paper–Scissors AI Game (Version 1 - Fast Reaction)")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Region of Interest
    roi = frame[100:350, 100:350]

    # Preprocess
    roi_resized = cv2.resize(roi, (128,128))
    roi_normalised = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_normalised, axis=0)

    # Prediction
    prediction = model.predict(roi_expanded)
    user_choice = labels[np.argmax(prediction)]

    ai_choice = random.choice(['rock', 'paper', 'scissors'])

    # Display result
    cv2.putText(frame, f"You: {user_choice}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"AI: {ai_choice}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.rectangle(frame, (100, 100), (350, 350), (255, 0, 0), 2)
    cv2.imshow("RPS AI Game - V1", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
