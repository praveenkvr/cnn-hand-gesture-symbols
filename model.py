from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, \
    Flatten, Dense, Dropout, BatchNormalization, LeakyReLU


def create_model(num_classes):
    """
    Create modal structure
    """
    model = Sequential([
        Conv2D(32, (3, 3), padding="same",
               activation="relu", input_shape=(64, 64, 1)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), padding="same"),
        LeakyReLU(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.7),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model
