import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def tune_model(X_train, X_test, y_train, y_test, seed=8, n_iter=30, test_size=0.2):

    param_dist = {
        "n_estimators":     randint(300, 2000),
        "max_depth":        randint(3, 10),
        "learning_rate":    loguniform(0.01, 0.3),
        "min_child_weight": randint(1, 20),
        "subsample":        uniform(0.6, 0.4),     # 0.6 to 1.0
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha":        loguniform(1e-3, 10),
        "reg_lambda":       loguniform(1e-3, 10),
        "gamma":            uniform(0, 5),
    }

    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, device="cuda"),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print(f"CV RMSE: {-search.best_score_:.4f}")
    print(f"best params: {search.best_params_}")

    train_pred = search.predict(X_train)
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")

    pred = search.predict(X_test)
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
    print(f"Test R2:   {r2_score(y_test, pred):.4f}")

    return search