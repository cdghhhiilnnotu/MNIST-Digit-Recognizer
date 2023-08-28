import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import math

model = keras.models.load_model('model-1.keras')

data_org = pd.read_csv('Data/train.csv')
data_org = data_org.values
data_org_X, data_org_Y = data_org[:,1:], data_org[:,0]
data_org_X = data_org_X / 255.

TRAIN_LIMIT = math.ceil(0.7 * data_org_X.shape[0])
VALID_LIMIT = TRAIN_LIMIT + math.ceil(0.2 * data_org_X.shape[0])
# TEST_LIMIT = math.ceil(0.1 * data_org_X.shape[0])

train_X, val_X, test_X = data_org_X[:TRAIN_LIMIT], data_org_X[TRAIN_LIMIT:VALID_LIMIT], data_org_X[VALID_LIMIT:]
train_Y, val_Y, test_Y = data_org_Y[:TRAIN_LIMIT], data_org_Y[TRAIN_LIMIT:VALID_LIMIT], data_org_Y[VALID_LIMIT:]

data =  pd.read_csv('Data/test.csv')
data = data.values

predictions = model.predict(data)
print(predictions)
print(np.argmax(predictions[227]))

test_img = data[227]
test_img = np.reshape(test_img, (28,28))

plt.imshow(test_img)
plt.show()

history = model.evaluate(test_X, test_Y)
acc_res = history.history['acc']
val_acc_res = history.history['val_acc']
loss_res = history.history['loss']
val_loss_res = history.history['val_loss']
print(history)








