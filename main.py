import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import math
import matplotlib.pyplot as plt

data = pd.read_csv('Data/train.csv')
data = data.values
data_X, data_Y = data[:,1:], data[:,0]
data_X = data_X / 255.

TRAIN_LIMIT = math.ceil(0.7 * data_X.shape[0])
VALID_LIMIT = TRAIN_LIMIT + math.ceil(0.2 * data_X.shape[0])
# TEST_LIMIT = math.ceil(0.1 * data_X.shape[0])

train_X, val_X, test_X = data_X[:TRAIN_LIMIT], data_X[TRAIN_LIMIT:VALID_LIMIT], data_X[VALID_LIMIT:]
train_Y, val_Y, test_Y = data_Y[:TRAIN_LIMIT], data_Y[TRAIN_LIMIT:VALID_LIMIT], data_Y[VALID_LIMIT:]

model = Sequential([
    InputLayer(input_shape=(data_X.shape[1])),
    Dense(784, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='softmax')
])
    
# model.add(InputLayer(input_shape=(data_X.shape[1])))
# model.add(Dense(784, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fitting = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=3, batch_size=50)

history = model.evaluate(test_X,test_Y)

model.save('model-batchsize-50.keras')

acc_res = fitting.history['accuracy']
val_acc_res = fitting.history['val_accuracy']
loss_res = fitting.history['loss']
val_loss_res = fitting.history['val_loss']

plt.plot(range(len(acc_res)),acc_res, label='Accuracy')
plt.plot(range(len(val_acc_res)),val_acc_res, label='Val_Accuracy')
plt.legend()
plt.show()

plt.plot(range(len(loss_res)),loss_res, label='Loss')
plt.plot(range(len(val_loss_res)),val_loss_res, label='Val_Loss')
plt.legend()
plt.show()

