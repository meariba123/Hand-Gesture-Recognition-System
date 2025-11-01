import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model 

#loads the fine tuned model
model = tf.keras.models.load_model('models/mobilenet_finetuned_week4.h5')

#this selects a sample image for visualisation 
img_path = 'dataset/val/rock/IMG_2132.jpg' #for testing purposes.
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0



#forces the model to build by calling it once
_ = model.predict(img_array)

if isinstance(model, tf.keras.Sequential):
    #finds the first nested Model layer (e.g., MobileNet)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            core_model = layer
            print(f"Unwrapped inner model: {core_model.name}")
            break
    else:
        core_model = model
else:
    core_model = model


#finds the last convolutional layer
last_conv_layer = None
for layer in core_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer

if last_conv_layer is None:
    raise ValueError("No Conv2D layer found. Grad-CAM requires a convolutional layer.")

print("Using last conv layer:", last_conv_layer.name)

#builds a model that outputs conv maps + model prediction
grad_model = Model(
    inputs=core_model.input,
    outputs=[last_conv_layer.output, core_model.output]
)

#predicts the class
preds = model.predict(img_array)
predicted_class = np.argmax(preds[0])
print("Predicted Class Index:", predicted_class)


#computes gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, predicted_class]

#gets the gradients of the target class
grads = tape.gradient(loss, conv_outputs)

#pools gradients over all axes except channels 
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
conv_outputs = conv_outputs[0]

#weights each channel by importance
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

#normalises between 0-1 for display
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

#loads original image for overlay
img_original = cv2.imread(img_path)
img_original = cv2.resize(img_original, (128, 128))

#converts heatmap to colour
heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#surperimpose heatmap onto original image
superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

#displays results
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()