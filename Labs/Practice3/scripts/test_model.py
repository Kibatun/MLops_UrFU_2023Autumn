import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
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
models_dir = base_dir / "models"

model = load_model(models_dir / "model.keras")

x_test = np.load(ds_dir / "x_test.npy")
y_test = np.load(ds_dir / "y_test.npy")


labels = [i for i in range(6)]
y_pred = model.predict(x_test, batch_size=16, verbose=1)
report = classification_report(
    np.argmax(y_test, axis=1),
    np.argmax(y_pred, axis=1),
    output_dict=True,
    target_names=labels
)
