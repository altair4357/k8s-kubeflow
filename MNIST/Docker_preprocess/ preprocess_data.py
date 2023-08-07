import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

def preprocess_data():
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)


if __name__ == '__main__':
    print("Preprocessing data. . .")
    preprocess_data()
   # x_train, y_train, x_test, y_test = preprocess_data()

   # np.save('x_train.npy', x_train)
   # np.save('y_train.npy', y_train)
   # np.save('x_test.npy', x_test)
   # np.save('y_test.npy', y_test)
