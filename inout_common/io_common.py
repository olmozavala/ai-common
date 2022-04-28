import os
from os.path import getmtime
from os import walk, listdir
from os.path import join
import pandas as pd

import numpy as np


def read_splits_file(file_name):
    """Reads a splits file and returns the ids for training, validation and test"""
    splits_df = pd.read_csv(file_name)
    train_ids = splits_df.iloc[:,0]
    val_ids = splits_df.iloc[:,1][splits_df.iloc[:,1] != -1]
    test_ids = splits_df.iloc[:,2][splits_df.iloc[:,2] != -1]

    return train_ids.values, val_ids.values, test_ids.values

