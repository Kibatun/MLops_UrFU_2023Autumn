"""
Некая предобработка
"""

import numpy as np
import pandas as pd
import pathlib

embeding = "longformer.npy"
labels = "target.npy"
base_dir = pathlib.Path().resolve()
ds_dir = base_dir / "datasets"


X = np.load(ds_dir / embeding)
# Y = np.load(ds_dir / labels)

# Добавляем третье измерения для lstm
X = np.expand_dims(X, 1)
np.save(ds_dir / "longformer_expanded.npy", X)
