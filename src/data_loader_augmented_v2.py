from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_augmented(data_dir, img_size=(128, 128), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8,1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, val_data