from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def MNISTExample(num_classes=10, input_shape=(28, 28)):
    """ A dummy sequential model. To be used to show the usage of the drawer function.
    # Returns
        A Keras model instance for training on the MNIST dataset.
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (*input_shape, 1)
    else:
        input_shape = (1, *input_shape)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
