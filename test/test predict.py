from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LSTM, Attention, Input, Layer
from keras.optimizers import Adam
import numpy as np
import time
import tensorflow

physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(9,)),
        Dense(32, activation='relu'),
        Dense(41)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model


model = create_model()
start = time.time()
for i in range(100):
    test_input = np.array(np.zeros(9).reshape(1, 9))
    model.predict(test_input)

    #print(model.predict(test_input))
stop = time.time()

print("time: ", stop - start)

