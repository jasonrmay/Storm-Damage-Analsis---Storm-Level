""" Logistic Regression Model for predicting IF a storm causes damage."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_logreg(X_train, X_test, y_train, y_test, seed=8):
    model = LogisticRegression(random_state=seed, max_iter=2500)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    print("Train Classification Report:")
    print(classification_report(y_train, train_pred))

    pred = model.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, pred))

    print(f"Test ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    return model