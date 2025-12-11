# Hand-Gesture-Recognition-System
### Deep Learning + Real-Time Interactive AI Gameplay

This project presents a **real-time hand gesture recognition system** that uses a **MobileNetV2 deep learning model** to classify images into the classes **Rock, Paper, Scissors, or None**, and then applies the model in an **AI-powered gameplay experience**.

The system includes:
-  **Fine-tuned MobileNetV2 model**
-  **Real-time webcam inference**
-  **RPS game against an AI opponent**
-  **Explainable AI (Grad-CAM visualisation)**
-  **Testing and evaluation with saved output results**

---

## Project Structure
<img width="596" height="902" alt="image" src="https://github.com/user-attachments/assets/43246067-180b-46d7-8d67-334831a3860c" />

## Dataset

The dataset was custom-captured and contains four gesture classes:

| Class | Description |
|-------|-------------|
| Rock | Closed fist |
| Paper | Flat open palm |
| Scissors | Two fingers |
| None | No valid gesture detected |

🔗 **Dataset Link:** *(Will be provided here)*

---

## Installation

### Clone project directory

```bash
git clone <https://github.com/meariba123/Hand-Gesture-Recognition-System.git>
cd HandGestureProject
```
---

### Install required libraries
```bash
pip install tensorflow opencv-python numpy matplotlib h5py
```

### Windows users only, for sound effects
```bash
pip install winsound
```
---

## Running the Applications
- Live Gesture Recognition
python src/app_realtime.py

### Rock–Paper–Scissors Game
Version 1 (Basic):
python src/rps_game_v1.py

Version 2 (Full Game + Testing + Saved Results)
python src/rps_game_v2.py

Explainable AI (Grad-CAM)
python src/visualise_model.py

---

## Testing & Performance Evaluation

Testing follows Criterion 4 of the assessment:

- Accuracy, F1-score & confusion matrix
- Error and misclassification analysis
- Explainable AI via Grad-CAM
- Real-world gameplay performance
- Efficiency comments & improvements

Testing outputs are automatically saved in /results/ when running:

python src/rps_game_v2.py

---

## AI transparency scale declaration statement

Artificial Intelligence (AI) has not been used for any part of the activity.

---

## Google Drive link to dataset

https://drive.google.com/drive/folders/1ireN1yHqNZSDfON9oSGoJJdDdz0_6iPd?usp=drive_link
