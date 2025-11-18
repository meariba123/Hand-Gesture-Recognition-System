import cv2
import numpy as np
import random
import time
from tensorflow.keras.models import load_model
import winsound  # Windows system beep

# Load model
model = load_model('models/mobilenet_finetuned_week4.h5')
labels = ['none', 'paper', 'rock', 'scissors']

# Webcam
cap = cv2.VideoCapture(0)

# Score tracking
user_score = 0
ai_score = 0
WINNING_SCORE = 5

print("Rock–Paper–Scissors AI Game starting... First to 5 wins!")
print("Press 'q' to quit.")

def get_winner(user, ai):
    if user == ai:
        return "Draw"
    elif (user == "rock" and ai == "scissors") or (user == "paper" and ai == "rock") or (user == "scissors" and ai == "paper"):
        return "You Win!"
    else:
        return "AI Wins!"

while user_score < WINNING_SCORE and ai_score < WINNING_SCORE:

    # Countdown
    for i in ["Get ready...", "3", "2", "1", "GO!"]:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, i, (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
        cv2.imshow("RPS AI Game", frame)

        if i in ["3", "2", "1"]:  # beep sound
            winsound.Beep(1000, 300)  # 1000 Hz for 0.3s
        elif i == "GO!":
            winsound.Beep(1500, 500)  # stronger sound for start

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # Capture final frame for prediction
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = frame[100:350, 100:350]

    # Preprocess
    roi_resized = cv2.resize(roi, (128,128))
    roi_normalised = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_normalised, axis=0)

    # Prediction
    prediction = model.predict(roi_expanded)
    user_choice = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    ai_choice = random.choice(['rock', 'paper', 'scissors'])
    result = get_winner(user_choice, ai_choice)

    if result == "You Win!":
        user_score += 1
    elif result == "AI Wins!":
        ai_score += 1

    # Display game result
    cv2.putText(frame, f"You: {user_choice} ({confidence:.1f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"AI: {ai_choice}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"Result: {result}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Score - You {user_score} : {ai_score} AI", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.rectangle(frame, (100, 100), (350, 350), (255, 0, 0), 2)

    cv2.imshow("RPS AI Game", frame)

    # Delay between rounds
    key = cv2.waitKey(2500)
    if key & 0xFF == ord('q'):
        break

# Final winner
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
if user_score == WINNING_SCORE:
    final_message = "Congratulations! You Won the Game 🎉"
else:
    final_message = "AI Won the Game 🤖 Better Luck Next Time"

cv2.putText(frame, final_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
cv2.imshow("RPS AI Game", frame)
cv2.waitKey(4000)

cap.release()
cv2.destroyAllWindows()
