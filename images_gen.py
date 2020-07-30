import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator


training_path = pathlib.Path('.').joinpath('data', 'training')
test_path = pathlib.Path('.').joinpath('data', 'test')

NUM_CLASSES = 5


def get_classes():
    return get_test_generator().class_indices


def get_index_class_mapping():
    return {v: k for k, v in get_classes().items()}


def get_train_generator():
    """
    Train Gen
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True,
    )

    return train_datagen.flow_from_directory(
        training_path, target_size=(64, 64), class_mode='categorical', batch_size=16, shuffle=True, color_mode='grayscale')


def get_test_generator():
    """
    Test gen
    """
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True
    )
    return test_datagen.flow_from_directory(
        test_path, target_size=(64, 64), class_mode='categorical', color_mode='grayscale')


if __name__ == '__main__':
    train_gen = get_train_generator()
    test_gen = get_test_generator()

    print(f"Classes {train_gen.num_classes}")
    print(f"Classes {test_gen.num_classes}")

    print(f"Samples {(train_gen.samples)}")
    print(f"Samples {(test_gen.samples)}")

    print("Class indicies", train_gen.class_indices)
    print(train_gen.class_mode)
