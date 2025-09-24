import cv2, os
import numpy as np

def load_images_from_folder(folder, label, size=(128,128)):
    data, labels = [], []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)
            data.append(img/255.0)
            labels.append(label)
    return np.array(data), np.array(labels)
