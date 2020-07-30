import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt
from util import plot_confusion_matrix

from model import create_model
from images_gen import get_test_generator, get_train_generator, get_classes
from sklearn.metrics import confusion_matrix

train_generator = get_train_generator()
test_generator = get_test_generator()
model = create_model(train_generator.num_classes)
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer='adam',  metrics=['accuracy'])

hist = model.fit(train_generator, steps_per_epoch=250 // 16, epochs=100,
                 verbose=2, validation_data=test_generator, validation_steps=1)


plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.title('loss')

y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, get_classes().values(), 'Confusion matrix')
plt.show()
model.save('hand_symbols.h5')
