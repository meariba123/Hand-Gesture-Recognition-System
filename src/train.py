# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from data_loader import load_images_from_folder
# from model_cnn import build_cnn

# # Example (fake data for now)
# import numpy as np
# X = np.random.rand(200,128,128,3)
# y = np.random.randint(0,4,200)

# # preprocess
# y = to_categorical(y, num_classes=4)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # build model
# model = build_cnn()

# # train
# model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_data
from model_cnn import create_cnn

#load dataset
train_data, val_data = load_data("dataset")

#build model
model=create_cnn(num_classes=3) #set to 3 at the moment as there is no data in "None"

#early stopping (what does early stopping mean)
early_stop=EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20, #what is epochs?
    callbacks=[early_stop]
)

#save model
model.save("models/cnn_week2.h5")

#plot accuracy
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy_curve.png")
plt.show()

#plot loss
plt.plot(history.history["loss"], label="Train Acc")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend() #what is legend
plt.savefig("results/loss_curve.png")
plt.show()