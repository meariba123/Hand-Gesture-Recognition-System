import cv2 #reference: https://pypi.org/project/opencv-python/#:~:text=cv2%20(old%20interface%20in%20old,of%20tutorials%20around%20the%20internet.
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#without cv2, the model would only work on saved images - but with OpenCV, it becomes interactive and real time 
#loads the fine tuned week 4 model
model = load_model('models/mobilenet_finetuned_week4.h5')

#define the class labels which is the same order as the dataset folders
labels = ['none', 'paper', 'rock', 'scissors']

#starts the webcam
cap = cv2.VideoCapture(0)

print('Real-time Hand Gesture Recognition started!')
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    #flips horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    #defining the region of interest (ROI) for gesture detection
    roi = frame[100:350, 100:350] #defines a cropped box for detecting the gesture 
    cv2.rectangle(frame, (100,100), (350,350), (0,255,0),2) #draws the green rectangle for the region interest (ROI)

    #preprocess ROI to match model input
    roi_resized = cv2.resize(roi, (128,128))
    roi_normalised = roi_resized.astype('float') / 255.0 #prepares the image to match the trained model input
    roi_expanded = np.expand.dims(roi_normalised, axis=0)

    #makes prediction
    prediction = model.predict(roi_expanded)
    class_index = np.argmax(prediction[0])
    label = labels[class_index]
    confidence = prediction[0][class_index] * 100

    #displays prediction
    cv2.putText(frame, f"{label} ({confidence:.1f}%)", (100, 90), #displays text like "rock (92%) on the video feed"
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    #shows webcam feed in a window 
    cv2.imshow("Hand Gesture Recognition", frame)

    #press q to quid
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
