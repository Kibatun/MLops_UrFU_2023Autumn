import numpy as np
import pathlib

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils


embeding = "longformer_expanded.npy"
labels = "target.npy"
base_dir = pathlib.Path().resolve()
ds_dir = base_dir / "datasets"


X = np.load(ds_dir / embeding)
Y = np.load(ds_dir / labels)


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

num_classes = len(np.unique(Y))
class_weights = compute_class_weight('balanced', classes=np.unique(Y), y=y_train)

y_train = utils.to_categorical(y_train, num_classes=num_classes, dtype='int64')
y_test = utils.to_categorical(y_test, num_classes=num_classes, dtype='int64')

np.save(ds_dir / "x_train.npy", x_train)
np.save(ds_dir / "x_test.npy", x_test)
np.save(ds_dir / "y_train.npy", y_train)
np.save(ds_dir / "y_test.npy", y_test)
