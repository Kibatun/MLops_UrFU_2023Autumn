"""
    Просто пример что в целом так можно но смысла не имеет, проще сразу открывать npy файлы
"""

import numpy as np
import pandas as pd
import pathlib

embeding = "longformer.npy"
labels = "target.npy"
base_dir = pathlib.Path().resolve()
ds_dir = base_dir / "datasets"

X = np.load(ds_dir / embeding)
Y = np.load(ds_dir / labels)

pd.DataFrame(X).to_csv(ds_dir / "data.csv")
pd.DataFrame(X).to_csv(ds_dir / "labels.csv")

