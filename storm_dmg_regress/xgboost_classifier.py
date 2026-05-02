from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

param_dist = {
    "n_estimators":     randint(300, 2000),
    "max_depth":        randint(3, 10),
    "learning_rate":    loguniform(0.01, 0.3),
    "min_child_weight": randint(1, 20),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha":        loguniform(1e-3, 10),
    "reg_lambda":       loguniform(1e-3, 10),
    "gamma":            uniform(0, 5),
}

def tune_log_model(X_train, X_test, y_train, y_test, seed=8, n_iter=30):
    search = RandomizedSearchCV(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=8,
        ),
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring="neg_log_loss",         # tuning for calibration, not just rank
        random_state=8,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    
    print(f"CV log loss:  {-search.best_score_:.4f}")
    print(f"best params:  {search.best_params_}")
    
    prob = search.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    
    print(f"holdout ROC AUC:    {roc_auc_score(y_test, prob):.4f}")
    print(f"holdout PR  AUC:    {average_precision_score(y_test, prob):.4f}")
    print(f"holdout log loss:   {log_loss(y_test, prob):.4f}")
    print(f"holdout Brier:      {brier_score_loss(y_test, prob):.4f}")
    print(f"holdout accuracy:   {(pred == y_test).mean():.4f}")

    return search