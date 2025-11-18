import cv2
import numpy as np
import random
import time
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import winsound
import os
import threading

# =========================
#      CONFIG SECTION
# =========================
MODEL_PATH = "models/mobilenet_finetuned_week4.h5"
RESULTS_DIR = "results"
WINNING_SCORE = 5
labels = ['none', 'paper', 'rock', 'scissors']
ai_choices = ['rock', 'paper', 'scissors']

# Create results folder
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Load model
model = load_model(MODEL_PATH)

# Webcam
cap = cv2.VideoCapture(0)

# Score tracking
user_score = 0
ai_score = 0

# For evaluation
true_labels = []
predicted_labels = []
latency_times = []

print("=== Rock–Paper–Scissors AI Game v2 Starting... ===")
print("➡ First to 5 wins! Press 'q' to quit.")

# Non-blocking beep
def beep(frequency, duration):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

# Determine winner
def get_winner(user, ai):
    if user == ai:
        return "Draw"
    elif (user == "rock" and ai == "scissors") or (user == "paper" and ai == "rock") or (user == "scissors" and ai == "paper"):
        return "You Win!"
    else:
        return "AI Wins!"

# ======================
#      GAME LOOP
# ======================
countdown_messages = ["Get ready...", "3", "2", "1", "GO!"]
COUNTDOWN_INTERVAL = 1.0  # seconds

while user_score < WINNING_SCORE and ai_score < WINNING_SCORE:
    countdown_index = 0
    countdown_time = time.time()
    round_done = False
    user_choice = "none"
    ai_choice = ""
    result = ""
    confidence = 0

    while not round_done:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Draw ROI
        cv2.rectangle(frame, (100,100), (350,350), (255,0,0), 2)

        # Update countdown every interval
        if countdown_index < len(countdown_messages) and time.time() - countdown_time >= COUNTDOWN_INTERVAL:
            msg = countdown_messages[countdown_index]
            if msg in ["3", "2", "1"]:
                beep(1000, 300)
            elif msg == "GO!":
                beep(1500, 500)

                # Capture ROI and predict
                start_time = time.time()
                roi = frame[100:350, 100:350]

                roi_resized = cv2.resize(roi, (128,128))
                roi_normalised = roi_resized / 255.0
                roi_expanded = np.expand_dims(roi_normalised, axis=0)

                prediction = model.predict(roi_expanded)
                user_choice = labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                latency = time.time() - start_time
                latency_times.append(latency)

                # AI choice and result
                ai_choice = random.choice(ai_choices)
                result = get_winner(user_choice, ai_choice)

                # Update scores
                if result == "You Win!":
                    user_score += 1
                elif result == "AI Wins!":
                    ai_score += 1

                # Testing metrics (model vs user)
                if user_choice != "none":
                    true_labels.append(user_choice)
                    predicted_labels.append(labels[np.argmax(prediction)])

            countdown_index += 1
            countdown_time = time.time()

        # Display countdown
        if countdown_index > 0:
            display_msg = countdown_messages[countdown_index-1]
            cv2.putText(frame, display_msg, (150,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)

        # Display scores & info
        cv2.putText(frame, f"You: {user_choice} ({confidence:.1f}%)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"AI: {ai_choice}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Result: {result}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Score - You {user_score} : {ai_score} AI", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if latency_times:
            cv2.putText(frame, f"Latency: {latency*1000:.1f} ms", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,100,100), 2)

        cv2.imshow("RPS AI Game v2", frame)

        # Move to next round 1 second after GO!
        if countdown_index >= len(countdown_messages) and time.time() - countdown_time >= 1.0:
            round_done = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            user_score = WINNING_SCORE  # force exit
            break

# ======================
#      FINAL RESULTS
# ======================
if true_labels:
    print("\n=== Saving Evaluation Results ===")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=ai_choices)
    with open(f"{RESULTS_DIR}/confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    print("[✔] Confusion Matrix saved")

    # Classification Report
    report = classification_report(true_labels, predicted_labels, labels=ai_choices)
    with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write(report)
    print("[✔] Classification Report saved")

    # Average latency
    avg_latency = sum(latency_times) / len(latency_times)
    with open(f"{RESULTS_DIR}/latency_analysis.txt", "w") as f:
        f.write(f"Average Real-time Prediction Latency: {avg_latency*1000:.2f}ms\n")
    print("[✔] Latency Analysis saved")
else:
    print("Not enough valid predictions for testing metrics.")

# Final Winner Display
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
if user_score == WINNING_SCORE:
    final_message = "Congratulations! You Won the Game."
else:
    final_message = "AI Won the Game. Better Luck Next Time!"

cv2.putText(frame, final_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
cv2.imshow("RPS AI Game v2", frame)
cv2.waitKey(4000)

cap.release()
cv2.destroyAllWindows()
print("\n=== Run Complete ===")
