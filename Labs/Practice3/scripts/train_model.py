import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import utils
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pathlib

base_dir = pathlib.Path().resolve()
ds_dir = base_dir / "datasets"


x_train = np.load(ds_dir / "x_train.npy")
y_train = np.load(ds_dir / "y_train.npy")


model = Sequential()
lstm_layer = LSTM(
    256,
    input_shape=(1, 768),
    dropout=0.1,
    activation="tanh"
)
model.add(lstm_layer)
model.add(Dense(6, activation="softmax"))

model.compile(optimizer="Adam", loss=CategoricalCrossentropy(), metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.3,
    # validation_data=(x_test, y_test),
)
model.save(base_dir / "models" / 'model.keras')
