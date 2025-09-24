from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from data_loader import load_images_from_folder
from model_cnn import build_cnn

# Example (fake data for now)
import numpy as np
X = np.random.rand(200,128,128,3)
y = np.random.randint(0,4,200)

# preprocess
y = to_categorical(y, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build model
model = build_cnn()

# train
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))
