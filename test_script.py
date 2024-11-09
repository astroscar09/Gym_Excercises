import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#import seaborn as sb

def read_csv(file):
    return pd.read_csv(file)

def copy_data(data):
    return data.copy()

def plot_data(x, y, kind = 'scatter'):
    fig, ax = plt.subplots()

    if kind == 'scatter':
        ax.scatter(x, y)
    elif kind == 'line':
        ax.plot(x, y)

    elif kind == 'bar':
        ax.bar(x, y)

    return fig, ax

def encode_data(data, col):
    
    data[col] = le.fit_transform(data[col])

    return data

def split_data(data, target_col, random_state, test_size = 0.75):
    
    X = data.drop(target_col, axis = 1)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators, max_depth, random_state):
    
    model = RandomForestClassifier(n_estimators = n_estimators, 
                                   max_depth = max_depth, 
                                   random_state = random_state)
    
    model.fit(X_train, y_train)

    return model
