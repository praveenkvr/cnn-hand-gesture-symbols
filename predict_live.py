import cv2
import numpy as np
from model import create_model
from images_gen import get_index_class_mapping, NUM_CLASSES
from util import morph_image

classes = get_index_class_mapping()


def predict_live(trained_model):
    """
    Captures live video and predicts based on the passed in
    model
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while cap.isOpened():

        ret, frame = cap.read()
        img = morph_image(frame)

        if not ret:
            continue

        # crop image to location of the square
        img_fr = img[10:320, 10:320]
        cv2.imshow('Cropped', img_fr)

        # resize and expand dims to (1, 64, 64, 1)
        img_fr = cv2.resize(img_fr, (64, 64))
        img_fr = np.expand_dims(img_fr, axis=0)
        img_fr = np.expand_dims(img_fr, axis=3)
        img_fr = img_fr / 255.
        result = trained_model.predict(img_fr)
        result = result > 0.9

        has_true = np.any(result, axis=1)
        result = result.argmax(axis=1)

        # update frame
        if has_true:
            cv2.putText(
                frame, f"result: {classes[result[0]]}", (
                    10, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA
            )
        cv2.rectangle(frame, (8, 8), (322, 322), (255, 0, 0), 1)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':

    model = create_model(NUM_CLASSES)
    model.load_weights('hand_symbols.h5')
    predict_live(model)
