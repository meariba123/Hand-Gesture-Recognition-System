import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from data_loader import load_data
from model_cnn import create_cnn



#load dataset
train_data, val_data = load_data("dataset")


print("Classes found:", train_data.class_indices)
print("Number of classes:", train_data.num_classes)

#build model
model=create_cnn(num_classes=4) 

#early stopping (what does early stopping mean)
early_stop=EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

csv_logger = CSVLogger(
    "results/training_log_cnn.csv",
    append=False
)


#train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20, #doing 20 complete passes so the model can learn and train the dataset
    callbacks=[early_stop, csv_logger]
)

#saving model
model.save("models/cnn_week2.h5")

#plotting accuracy
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend() #helps with visualisation
plt.savefig("results/accuracy_curve_cnn.png") #saving the figure into this file
plt.show() #shows the fgiure on display 

#plotting loss
plt.plot(history.history["loss"], label="Train Acc")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend() 
plt.savefig("results/loss_curve_cnn.png")
plt.show()