from tensorflow.keras import layers, models

#needs explaining 
def create_cnn(num_classes=3): #only 3 for now because "None" has no data
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
