# model training logic goes here
# How can we train test split the data?
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vnstock import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# where is df defined?
# we can assume df is the DataFrame returned from feature building and label generation
# e.g. df = build_features() merged with generate_labels()
# for simplicity, let's assume df is already defined here
# df = ...build_features()
# labels = generate_labels()
# df = df.merge(labels, left_index=True, right_index=True)  
# can we directly use df here from build_features.py?
# yes, we can import build_features from build_features.py and generate_labels from generate_label.py
#from features.build_features import build_features
#from src.labels.generate_label import generate_labels

import sys
print(sys.path)
#raise RuntimeError(sys.path)

#print(vnstock_trade.__file__)


from vnstock_trade.labels.generate import generate_labels
import os


def train():
    # Do higher predicted labels actually lead to higher future returns?
    features_df = pd.read_csv(os.path.join('./data/VNM_2022.csv'))
    labels = generate_labels(features_df)
    
    # doing a merge where both DataFrames contain columns with the same names, pandas auto-renames them to avoid overwriting
    # df = features_df.merge(labels, left_index=True, right_index=True)
    df = features_df.merge(labels)
    data = df.dropna() 
    
    print(f"columns: {data.columns}")

    feature_cols = ["return_1", "log_return_1", "ma_5", "return_5", "volatility_5", "lag_logret_1", "lag_logret_2", "lag_logret_3", "lag_logret_4", "lag_logret_5", "mom_5", "vol_change", "vol_ma_5", "rsi_14", "macd"]

    X = data[feature_cols]
    print(f"X feature cols: {X.columns}")
    y = data['label']

    # Time-based split, 70% for training
    split = int(len(data) * 0.7)
    print(f"Data split at index: {split}, total length: {len(data)}")

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]


    model = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_split=20,random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values(by='importance', ascending=False)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Feature Importances:")
    print(feature_importance_df)

    # Do higher predicted labels actually lead to higher future returns?
    test_results = data[split:].copy()
    test_results['predicted_label'] = y_pred
    print(f"Average returns by predicted label: {test_results.groupby('predicted_label')['return_t+N'].mean()}")
    # Seeing Return by class is incomplete -> we also need volatility of those returns, stability across time
    # bc High mean + huge volatility = unstable strategy, Financial ML is about: Maximizing signal-to-noise ratio, Not maximizing accuracy.
    # print results to see return_t+N to compare with real returns
    base = os.getcwd()
    out_path = os.path.join(base, 'data', 'test_results.csv')
    test_results.to_csv(out_path, index=False)
    print(f"Test Results output to {out_path}")
    
def compute_slope(window):
    x = np.arange(len(window)) # time is evenly spaced, in this case we want to create x coordinates to pair with y values in window, then we can find best fit line
    if len(x) < 2:
        return 0  # Not enough points to compute slope
    slope = np.polyfit(x, window, 1)[0]  # Get the slope (coefficient of x)
    return slope

def compute_ma(window):
    return np.mean(window)



def _train():
    """_summary_: this computes 
    
    """
    data = pd.read_csv(os.path.join('./data/test_results.csv')) #features_VNM_2022
    print(data)
    
    df = data[['time', 'close']].dropna()
    
    df.reset_index(drop=True, inplace=True)
    
    prices = df['close'].values
    print(prices)
    
    window = 5
    
    X = []
    y = []
    
    # take window(number)(, i.e 5) recent prices starting at postition i
    """
    for i in range(len(prices) - window - 1):
        w = prices[i:i+window]
        print(f"w: {w}")
        slope = compute_slope(w)
        print(f"slope: {slope}")
        ma = compute_ma(w)
        
        current_price = prices[i+window-1]
        #Features 
        X.append([slope, current_price - ma]) # distance from moving average as a feature, if price is above ma, it may indicate upward momentum, if below, downward momentum
        #Label: whether price goes up or down the next day
        next_price = prices[i+window]
        y.append(1 if next_price > current_price else 0)
    """
    for i in range(len(prices) - window - 6):
        w = prices[i:i+window]
        z = (w[-1] - np.mean(w)) / np.std(w)
        dist = (w[-1] - np.mean(w)) / np.mean(w)
        momentum = w[-1] - w[-2]
        vol = np.std(w)

        X.append([z, dist, momentum, vol])
        # redefine label
        future_return = (prices[i+3] - prices[i]) / prices[i]
        y.append(1 if future_return > 0 else 0)
    
    X = np.array(X)
    print("len X:", len(X))
    y = np.array(y)
    print(f"X: {X[:20]}, y: {y[:20]}")  # Print first 5 samples for sanity check
    
    # results show that there is a lag between prices, for example, the distance from ma feature reacts to price changes with a delay, which is expected since it's based on past prices, this lag can make it challenging for the model to capture sudden price movements,
    # which are common in stock markets, and may require more sophisticated features or models that can capture temporal dependencies better, such as LSTM or Transformer-based models.
    # the lag is basically this: if the distance is positive before price decreasess, price already started to decrease but the feature still shows positive distance until the price crosses below the ma, which creates a lag in the feature's response to price changes, this is a common issue with technical indicators that are based on past prices, and it highlights the importance of feature engineering and selection in financial machine learning, as well as the potential need for models that can capture temporal dependencies more effectively.
    # if the distance from ma is negative before price increases, prices already started to decrease, reflecting Vietnam market's tendency of overreacting to bad news and underreacting to good news, which creates a lag in the feature's response to price changes, this is a common issue with technical indicators that are based on past prices, and it highlights the importance of feature engineering and selection in financial machine learning, as well as the potential need for models that can capture temporal dependencies more effectively.
    # how can  
    
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = LogisticRegression()    
    
    model.fit(X_train, y_train)
    
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_test)
        
    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_test, val_probs)
    
    print("Train loss:", train_loss)
    print("Validation loss:", val_loss)
     
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
     
    # Random prediction
    random_preds = np.random.randint(0, 2, size=len(y_test))
    print("Random accuracy:", accuracy_score(y_test, random_preds))
   
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)

    model.fit(X_train, y_shuffled[:split])
    y_pred_fake = model.predict(X_test)

    print("Accuracy with shuffled labels:", accuracy_score(y_test, y_pred_fake))
    
    
    # Walk-forward accuracy
    accuracies = []
    
    print("len x: ", len(X))

    for i in range(50, len(X)-1):
    
        X_train = X[:i]
        y_train = y[:i]
    
        X_test = X[i:i+1]
        y_test = y[i:i+1]
    
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
    
        accuracies.append(int(pred[0] == y_test[0]))

    #print(f"accuracies: {accuracies}")

    print("Walk-forward accuracy:", np.mean(accuracies))
    
    # Backtest
    capital = 1.0

    for i in range(split, len(X)-1):
    
        X_train = X[:i]
        y_train = y[:i]
    
        model.fit(X_train, y_train)
    
        prob = model.predict_proba([X[i]])[0][1]
    
        if prob > 0.6:
            ret = (prices[i+1] - prices[i]) / prices[i]
            capital *= (1 + ret)

    print("Final capital:", capital)
    
    # model features mix trend signal(slope) and reversal signal(distance) -> conflicting signals
    # market behaves more like mean-reversion than trend-following   
    # class imbalance too
    # Feature	             Meaning
    # very high (above MA)	 overbought → likely go DOWN
    # very low (below MA)	 oversold → likely go UP
if __name__ == "__main__":
    #train()
    _train()



