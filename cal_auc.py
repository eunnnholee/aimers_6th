import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cal_auc.py가 있는 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")


def calculate_auc(y_pred, seed=1):
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    train, test = train_test_split(train_data, test_size=0.2, random_state=seed, stratify=train_data['임신 성공 여부'])
    y_test = test['임신 성공 여부'].copy()
    auc = roc_auc_score(y_test, y_pred)
    return auc



