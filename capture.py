import pathlib
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', choices=['training', 'test'])
parser.add_argument(
    '-n', '--name', help="enter name of folder", required=True, type=str)
parser.add_argument('-g', '--grayscale',
                    help="convert images to greyscale", action="store_true")
args = parser.parse_args()


def generate_images(name, grayscale=False, mode='training'):
    """
    Generate images from webcam
    """
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    count = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # draw rectangle
        cv2.rectangle(frame, (98, 98), (422, 422), (255, 0, 0), 1)
        img = frame
        if grayscale:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        cv2.imshow('Image', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            crop = img[100:420, 100:420]
            pathlib.Path(
                f'./data/{mode}/{name}').mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f'./data/{mode}/{name}/{count}.jpg', crop)
            count += 1
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_images(args.name, args.grayscale, args.mode)
