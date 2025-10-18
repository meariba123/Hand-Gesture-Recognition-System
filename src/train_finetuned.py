import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping 
from data_loader_augmented_v2 import load_data_augmented
from model_transfer_finetuned import create_finetuned_model

#loading the dataset
train_data, val_data = load_data_augmented("dataset")

#builds the fine tuned model
model = create_finetuned_model(num_classes=4)

#early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

#training model
history=model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop]
)

#saving model
model.save('models/mobilenet_finetuned_week4.h5')

#plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/accuracy_curve_week4.png')
plt.show()

#plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/loss_curve_week4.png')
plt.show()