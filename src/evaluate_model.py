import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns #helps explore the data from each of the models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#loading the trained model
model = load_model('models/mobilenet_finetuned_week4.h5')

print("Total parameters (Transfer Model):", model.count_params())

#loading the validation dataset
data_dir = 'dataset'
img_size = (128, 128)
batch_size = 32

val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    f"{data_dir}/val",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False #keeping the order for correct labels
)

start = time.time()
predictions = model.predict(val_data)
end = time.time()

print("Trained Model inference time (seconds):", end - start)
print("Average time per image:", (end - start) / val_data.samples)


#predictions on the validation data
predictions = model.predict(val_data)
predicted_classes = np.argmax(predictions, axis=1) #reference: https://www.analyticsvidhya.com/blog/2023/12/all-you-need-to-know-about-numpys-argmax-function/
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys()) #typo on indicies -  should be indices

#plotting classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:\n")
print(report)

#plotting confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title("Confusion Matrix - Fine-tuned Model (Week 5)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/confusion_matrix_week5.png")
plt.show()


#per-class accuracy - shows how well each gesture is recognised
class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\nPer-Class Accuracy:")
for label, acc in zip(class_labels, class_accuracy):
    print(f"{label}: {acc:.2f}")

#explicit misclassification counts - helps identify which classes are confused
#https://www.enki.com/post/what-does-enumerate-mean-in-python#:~:text=The%20enumerate()%20function%20is,need%20to%20manually%20track%20indexes.
print("\nMisclassification Analysis:")
for i, true_label in enumerate(class_labels):
    for j, pred_label in enumerate(class_labels):
        if i != j and cm[i, j] > 0:
            print(f"{true_label} misclassified as {pred_label}: {cm[i, j]} times")


#confidence analysis (model uncertainty) - used to justify improvement strategies
confidence_scores = np.max(predictions, axis=1)

print("\nPrediction Confidence Analysis:")
print("Average confidence:", np.mean(confidence_scores))
print("Lowest confidence:", np.min(confidence_scores))