import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator


training_path = pathlib.Path('.').joinpath('data', 'training')
test_path = pathlib.Path('.').joinpath('data', 'test')


def get_train_generator():
    """
    Train Gen
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True,
    )

    return train_datagen.flow_from_directory(
        training_path, target_size=(64, 64), class_mode='categorical', batch_size=16)


def get_test_generator():
    """
    Test gen
    """
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    return test_datagen.flow_from_directory(
        test_path, target_size=(64, 64), class_mode='categorical')


if __name__ == '__main__':
    train_gen = get_train_generator()
    test_gen = get_test_generator()

    print(f"Classes {train_gen.num_classes}")
    print(f"Classes {test_gen.num_classes}")

    print(f"Samples {(train_gen.samples)}")
    print(f"Samples {(test_gen.samples)}")

    print("Class indicies", train_gen.class_indices)
    print(train_gen.class_mode)
