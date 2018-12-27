import numpy as np
import keras.backend as K

from keras.datasets import mnist
from keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

import random
random.seed(10000)
index = [i for i in range(x_test.shape[0])]
random.shuffle(index)
x_test = x_test[index]
y_test = y_test[index]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from keras.models import load_model
model = load_model('/home/wyh/PycharmProjects/DNN_testing/model/lenet5.h5')
x_test = x_test[:100]
label = np.argmax(model.predict(x_test), axis=1)
input_tensor = model.input
from keras.utils.np_utils import to_categorical
target = to_categorical(label, num_classes=model.output_shape[1])

output = model.layers[-1].output
fun = K.function([input_tensor], [output])
loss = K.categorical_crossentropy(target, output, from_logits=False)
grads = K.gradients(loss, input_tensor)[0]
iterate = K.function([input_tensor], [loss, grads])

loss, grads = iterate([x_test])
normalized_grad = np.sign(grads)
eps = 0.1
adv = np.clip(x_test + eps * normalized_grad, 0, 1)

import imageio
import time
for i in range(adv.shape[0]):
    label = np.argmax(y_test[i])
    img = adv[i]
    pred = np.argmax(model.predict(img.reshape(1, 28, 28, 1))[0])
    if label != pred:
      img *= 255
      img = img.astype('uint8')
      imageio.imwrite("/home/wyh/PycharmProjects/DNN_testing/results/fgsm/fgsm_" + str(pred)
                      + "_" + str(time.clock()) + "_" + str(label) +".png",
             img.reshape(28, 28))

