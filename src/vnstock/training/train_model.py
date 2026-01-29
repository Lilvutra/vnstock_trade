# model training logic goes here
# How can we train test split the data?
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# where is df defined?
# we can assume df is the DataFrame returned from feature building and label generation
# e.g. df = build_features(market_df) merged with generate_labels(market_df)
# for simplicity, let's assume df is already defined here
# df = ...build_features(market_df)
# labels = generate_labels(market_df)
# df = df.merge(labels, left_index=True, right_index=True)  
# can we directly use df here from build_features.py?
# yes, we can import build_features from build_features.py and generate_labels from generate_label.py
#from features.build_features import build_features
#from src.labels.generate_label import generate_labels
from pathlib import Path

from vnstock.labels.generate_label import generate_labels

from vnstock.features.build_features import build_features

import pandas as pd
import numpy as np 
import os
import sys

def train():
    # Do higher predicted labels actually lead to higher future returns?
    market_df = pd.read_csv(os.path.join('./data/features_VNM_2022.csv'))
    # features = build_features(market_df)
    labels = generate_labels(market_df)
    df = market_df.merge(labels, left_index=True, right_index=True)
    data = df.dropna() 

    feature_cols = ["return_1", "log_return_1", "ma_5", "return_5", "volatility_5", "lag_logret_1", "lag_logret_2", "lag_logret_3", "lag_logret_4", "lag_logret_5", "mom_5", "vol_change", "vol_ma_5", "rsi_14", "macd"]

    X = df[feature_cols]
    y = df['label']

    split = int(len(df) * 0.7)

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]


    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train()



