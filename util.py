import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt


def morph_image(frame):
    """
    Source: https://stackoverflow.com/questions/60759031/computer-vision-creating-mask-of-hand-using-opencv-and-python

    Masks the original image to a b&w image
    """
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    img = cv2.inRange(img, (0, 50, 20), (255, 255, 255))
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    return img


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix'):
    """
    source: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = cm.get_cmap("Spectral")
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
