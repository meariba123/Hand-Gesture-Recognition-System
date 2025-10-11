#this will be used for week 3 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_transfer_model(num_classes=4):
    base_model=MobileNetV2(
        input_shape=(128,128,3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable=False

    model = model.sSequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='catgorical_crossentropy',
        metrics=['accuracy']
    )

    return model