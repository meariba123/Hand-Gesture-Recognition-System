from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_finetuned_model(num_classes=4):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))

    #freezing most layers, unfreeze last 20 for fine tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model