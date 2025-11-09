import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

# Load your Week 4 fine-tuned model
model = load_model('models/mobilenet_finetuned_week4.h5')

# The same class order you used during training
labels = ['none', 'paper', 'rock', 'scissors']

# Function to decide winner
def get_winner(user, ai):
    if user == ai:
        return "Draw"
    elif (user == "rock" and ai == "scissors") or (user == "paper" and ai == "rock") or (user == "scissors" and ai == "paper"):
        return "You Win!"
    else:
        return "AI Wins!"

# Start webcam
cap = cv2.VideoCapture(0)
print("Rock–Paper–Scissors AI Game Started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:350, 100:350]
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (128, 128))
    roi_normalized = roi_resized.astype('float') / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Predict gesture
    prediction = model.predict(roi_expanded)
    class_index = np.argmax(prediction[0])
    user_choice = labels[class_index]

    # AI randomly selects a move
    ai_choice = random.choice(['rock', 'paper', 'scissors'])

    # Only decide winner if the model detects a valid gesture
    if user_choice != 'none':
        result = get_winner(user_choice, ai_choice)
    else:
        result = "Show a hand gesture!"

    # Display info on screen
    cv2.putText(frame, f'You: {user_choice}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f'AI: {ai_choice}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f'Result: {result}', (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Rock–Paper–Scissors AI Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
