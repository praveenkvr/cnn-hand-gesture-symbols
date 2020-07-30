import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_model
from images_gen import get_test_generator, NUM_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='pass image to predict')
args = parser.parse_args()

class_indices = get_test_generator().class_indices
result_mapping = {v: k for (k, v) in class_indices.items()}


def predict(img_path):
    """
    predict using loaded modal
    """
    model = create_model(NUM_CLASSES)
    model.load_weights('hand_symbols.h5')

    image = load_img(img_path, target_size=(64, 64), color_mode="grayscale")
    img_arr = img_to_array(image)
    img_arr = img_arr / 255.
    img_arr = np.expand_dims(img_arr, axis=0)
    res = model.predict(img_arr)
    print(res)
    print(np.argmax(res))
    print(f"Result is: {result_mapping[np.argmax(res)]}")


if __name__ == '__main__':
    predict(args.image)
