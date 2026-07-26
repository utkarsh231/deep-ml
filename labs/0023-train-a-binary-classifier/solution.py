import numpy as np
# You can import any sklearn module you need
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function

    model = LogisticRegression(max_iter = 1000)

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_predictions)

    def predict(X):
        return model.predict(X)

    return predict
