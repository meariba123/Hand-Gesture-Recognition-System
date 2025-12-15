# Hand Gesture Recognition System  
### Deep Learning, Transfer Learning & Interactive AI Gameplay

This project implements a **real-time hand gesture recognition system** for the game **Rock–Paper–Scissors**, developed using **deep learning techniques**.  
The system combines a **custom-built convolutional neural network (CNN)**, **transfer learning using MobileNetV2**, and **interactive AI gameplay**, demonstrating both fundamental and advanced AI concepts.

The solution was designed and implemented in full accordance with the assessment brief, covering **dataset creation**, **model training**, **performance evaluation**, and **additional AI capabilities**.

---

## Key Features

- Custom image dataset captured using a webcam
- CNN model built **from scratch**
- Transfer learning using **MobileNetV2**
- Fine-tuned pre-trained model for improved accuracy
- Real-time webcam-based gesture recognition
- Interactive Rock–Paper–Scissors gameplay
- Smart AI opponent with predictive behaviour
- Explainable AI using **Grad-CAM**
- Training logs, evaluation metrics, and result visualisations

---

## Dataset

The dataset was **entirely self-created** using a webcam and contains four gesture classes:

| Class | Description |
|------|------------|
| Rock | Closed fist |
| Paper | Open palm |
| Scissors | Two fingers |
| None | No valid hand gesture |

- Minimum of **100 images per class**
- Includes variation in hand position, orientation, and lighting
- Noise and background variation added for robustness
- Images resized to a consistent resolution

📂 **Dataset Access (Google Drive):**  
https://drive.google.com/drive/folders/1ireN1yHqNZSDfON9oSGoJJdDdz0_6iPd

---

## File Organisation

```text
├── dataset/
│   ├── train/
│   ├── val/

├── models/
│   ├── cnn_week2.h5
│   └── mobilenet_finetuned_week4.h5

│
├── results/
│   ├── accuracy_curve_cnn.png
│   ├── accuracy_curve_finetuned.png 
│   ├── accuracy_curve_week2.png 
│   ├── classification_report.txt
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_week5.png
│   ├── confusion_matrix.txt
│   ├── gradcam_output.jpg
│   ├── latency_analysis.txt
│   ├── loss_curve_cnn.png
│   ├── loss_curve_finetuned.png
│   ├── loss_curve_week2.png
│   ├── training_log_cnn.csv
│   ├── training_log_finetuned.csv
│   ├── training_log_transfer.csv
│   ├── transfer_accuracy_week3.png
│   ├── transfer_accuracy.png
│   ├── transfer_loss_week3.ong
│   ├── transfer_loss.png

├── src/
|   ├── app_realtime.py            # Real-time webcam inference
|   ├── data_loader_augmented_v2.py
|   ├── data_loader_augmented.py
|   ├── data_loader.py
|   ├── evaluate_model_cnn.py
|   ├── evaluate_model.py
|   ├── model_cnn.py
|   ├── model_transfer_finetuned.py
|   ├── model_transfer.py
│   ├── rps_game_v1.py             # Basic RPS gameplay
│   ├── rps_game_v2.py             # Smart AI gameplay + evaluation
│   ├── train_transfer.py          # Transfer learning (MobileNetV2 frozen)
│   ├── train_finetuned.py         # Fine-tuned MobileNetV2
│   ├── train.py                   # CNN model training (from scratch)
│   └── visualise_model.py         # Grad-CAM explainability
│
└── README.md
└── requirements.txt
```


---

## Installation

### Clone project directory

```bash 
git clone https://github.com/meariba123/Hand-Gesture-Recognition-System.git
cd Hand-Gesture-Recognition-System
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

## Models & Training Strategy

Three model configurations were implemented and evaluated to demonstrate progressive improvement and understanding of deep learning concepts:

### 1. Custom CNN (From Scratch)
- Implemented using TensorFlow/Keras
- Trained on the custom gesture dataset
- Serves as a baseline model
- Demonstrates understanding of convolutional layers, pooling, and dense classification

Training was computationally expensive, with each epoch taking approximately 2 minutes.  
For this reason, full training was completed offline and results were saved as logs and plots.

---

### 2. Transfer Learning (MobileNetV2 – Frozen Base)
- Pre-trained MobileNetV2 model loaded with ImageNet weights
- Feature extraction only (base layers frozen)
- Faster training time
- Improved convergence compared to the custom CNN

This model was selected for **live demonstration** due to its fast training and inference speed.

---

### 3. Fine-Tuned MobileNetV2
- Upper layers of MobileNetV2 unfrozen
- Fine-tuning performed with data augmentation
- Achieved the best validation performance
- Balanced accuracy and generalisation

The fine-tuned model is used in the final real-time application and gameplay system.

---

## Training Logs & Saved Results

Due to time and computational constraints, model training was completed prior to the video demonstration.

All training sessions automatically generate:
- CSV logs (epoch, accuracy, loss, validation metrics)
- Accuracy and loss curves
- Saved trained models

These files are stored in the `/results/` and `/models/` directories and are referenced during the video walkthrough to demonstrate learning behaviour and model performance.

---

## Real-Time System Behaviour

The real-time application uses:
- OpenCV for webcam capture
- Frame-by-frame inference
- Confidence thresholding to reduce false predictions
- Class stabilisation logic to improve gameplay reliability

Latency testing was conducted, and results are documented in: results/latency_analysis.txt

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

AITS Level: 1 – No AI

Artificial Intelligence (AI) has not been used to generate or author the submitted code, documentation, or experimental results.
All implementation, experimentation, evaluation, and analysis were carried out independently by the student.

---

