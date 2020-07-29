from tensorflow.keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt

from model import create_model
from images_gen import get_test_generator, get_train_generator


train_generator = get_train_generator()
test_generator = get_test_generator()
model = create_model(train_generator.num_classes)
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer='rmsprop', metrics=['accuracy'])

hist = model.fit_generator(train_generator, steps_per_epoch=250 // 16, epochs=100,
                           verbose=2, validation_data=test_generator, validation_steps=1)


plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.title('loss')

plt.show()
model.save('hand_symbols.h5')
