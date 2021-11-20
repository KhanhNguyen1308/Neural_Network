import cv2
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
print("Training? Y/N")
answer = input()
if answer == "Y" or answer == "y":
    data_dir = pathlib.Path("data")
    image_count = len(list(data_dir.glob('*/*.jpg')))
    batch_size = 16
    img_height = 224
    img_width = 224
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)
    num_classes = len(class_names)
    model = Sequential([
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # model.compile(
    #   optimizer='adam',
    #   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    #   metrics=['accuracy'])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=5
    )

    model.summary()
    # keras.utils.plot_model(model, show_shapes=True)
    model.save("model.h5")
else:
    model = load_model("model.h5")
img = cv2.imread("face_test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
print(predictions*100)

