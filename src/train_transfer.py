#week 3
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from src.data_loader_augmented import load_data_augmented
from src.model_transfer import create_transfer_model

#load dataset
train_data, val_data = load_data_augmented("dataset")

#build transfer model
model=create_transfer_model(num_classes=4)

early_stop=EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#train mdoel
history=model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

#save model
model.save('models/transfer_week3.h5')

#plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/accuracy_curve_week3.png')
plt.show()

#plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/loss_curve_week3.png')
plt.show()