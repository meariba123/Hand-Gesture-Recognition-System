import cv2
import numpy as np
import random
import time
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import winsound
import os
import threading
from collections import Counter


MODEL_PATH = "models/mobilenet_finetuned_week4.h5"
RESULTS_DIR = "results"
WINNING_SCORE = 5
labels = ['none', 'paper', 'rock', 'scissors']
ai_choices = ['rock', 'paper', 'scissors']

CONFIDENCE_THRESHOLD = 60

user_score = 0
ai_score = 0
human_history = []

#creates results folder
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

#loads model
model = load_model(MODEL_PATH)

#webcam
cap = cv2.VideoCapture(0)

#score tracking
user_score = 0
ai_score = 0

#for evaluation
true_labels = []
predicted_labels = []
latency_times = []

print("Rock–Paper–Scissors AI Game v2 Starting... ")
print("➡ First to 5 wins! Press 'q' to quit.")

#non-blocking beep
def beep(frequency, duration):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

#determines winner
def get_winner(user, ai):
    if user == ai:
        return "Draw"
    elif (user == "rock" and ai == "scissors") or (user == "paper" and ai == "rock") or (user == "scissors" and ai == "paper"):
        return "You Win!"
    else:
        return "AI Wins!"


#game loop
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

        #draws ROI
        cv2.rectangle(frame, (100,100), (350,350), (255,0,0), 2)

        #updates countdown every interval - reference: https://stackoverflow.com/questions/23190439/creating-a-count-down-timer-in-python-opencv
        if countdown_index < len(countdown_messages) and time.time() - countdown_time >= COUNTDOWN_INTERVAL:
            msg = countdown_messages[countdown_index]
            if msg in ["3", "2", "1"]:
                beep(1000, 300)
            elif msg == "GO!":
                beep(1500, 500)

                #captures ROI and predict
                start_time = time.time()
                roi = frame[100:350, 100:350]

                roi_resized = cv2.resize(roi, (128,128))
                roi_normalised = roi_resized / 255.0
                roi_expanded = np.expand_dims(roi_normalised, axis=0)

                prediction = model.predict(roi_expanded)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index] * 100

                if confidence < CONFIDENCE_THRESHOLD:
                    user_choice = "none"
                else:
                    user_choice = labels[pred_index]

                confidence = np.max(prediction) * 100
                latency = time.time() - start_time
                latency_times.append(latency)

                #AI choice and result
                #stores valid human moves
                if user_choice in ai_choices:
                    human_history.append(user_choice)

                #smart AI decision based on last 5 moves
                if len(human_history) >= 3:
                    recent_moves = human_history[-5:]
                    most_common = Counter(recent_moves).most_common(1)[0][0]

                    #AI plays the counter move
                    if most_common == "rock":
                        ai_choice = "paper"
                    elif most_common == "paper":
                        ai_choice = "scissors"
                    else:
                        ai_choice = "rock"
                else:
                    ai_choice = random.choice(ai_choices)

                #determine round winner
                result = get_winner(user_choice, ai_choice)

                #updates scores
                if result == "You Win!":
                    user_score += 1
                elif result == "AI Wins!":
                    ai_score += 1

                #testing metrics (model vs user)
                if user_choice != "none":
                    true_labels.append(user_choice)
                    predicted_labels.append(labels[np.argmax(prediction)])

            countdown_index += 1
            countdown_time = time.time()

        #displays countdown for users 
        if countdown_index > 0:
            display_msg = countdown_messages[countdown_index-1]
            cv2.putText(frame, display_msg, (150,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)

        #displays scores & info - references: https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/
        cv2.putText(frame, f"You: {user_choice} ({confidence:.1f}%)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"AI: {ai_choice}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Result: {result}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Score - You {user_score} : {ai_score} AI", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if latency_times:
            cv2.putText(frame, f"Latency: {latency*1000:.1f} ms", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,100,100), 2)

        cv2.imshow("RPS AI Game v2", frame)

        #moves to next round 1 second after GO!
        if countdown_index >= len(countdown_messages) and time.time() - countdown_time >= 1.0:
            round_done = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            user_score = WINNING_SCORE  #force exit
            break

#final results
if true_labels:
    print("\nSaving Evaluation Results ")
    
    #confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=ai_choices)
    with open(f"{RESULTS_DIR}/confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    print("Confusion Matrix saved")

    #classification Report
    report = classification_report(true_labels, predicted_labels, labels=ai_choices)
    with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write(report)
    print("[✔] Classification Report saved")

    #average latency
    avg_latency = sum(latency_times) / len(latency_times)
    with open(f"{RESULTS_DIR}/latency_analysis.txt", "w") as f:
        f.write(f"Average Real-time Prediction Latency: {avg_latency*1000:.2f}ms\n")
    print("Latency Analysis saved")
else:
    print("Not enough valid predictions for testing metrics.")

#final winner display
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
print("\n Run Complete ")
