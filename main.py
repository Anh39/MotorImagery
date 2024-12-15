import json
import biosig
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Literal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
from wave_dataloader import MotorImageryDataset, EVENT_TYPES, Util
import torch
import torch.utils
import torch.utils.data

dataloader = Util.read_single_data("B0101T.gdf", n_workers=0, bsz=2)
for item in dataloader:
    item_data, item_label = item
    print(item_data.shape)
    print(item_label.shape)
    item_data = item_data[0].T[0]
    item_label = item_label[0]
    plt.figure(figsize=(15,5))
    plt.plot(item_data)
    break