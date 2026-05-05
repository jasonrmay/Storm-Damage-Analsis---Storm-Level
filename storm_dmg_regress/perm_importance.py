from sklearn.inspection import permutation_importance
import pandas as pd

def perm_imp(X_train, y_train, X_test, y_test, model):
    from sklearn.inspection import permutation_importance
    result_train = permutation_importance(model.best_estimator_, X_train, y_train, n_repeats=10, random_state=8, n_jobs=-1, scoring="neg_root_mean_squared_error")
    result_test = permutation_importance(model.best_estimator_, X_test, y_test, n_repeats=10, random_state=8, n_jobs=-1, scoring="neg_root_mean_squared_error")
    return result_train, result_test

def perm_imp_collapse(result, X, dummy_prefixes = []):
    
    perm_imp = pd.Series(
        result.importances_mean, index=X.columns,
    ).sort_values(ascending=False)

    if len(dummy_prefixes) == 0:
        return perm_imp

    perm_imp_collapsed = perm_imp.copy()
    for prefix in dummy_prefixes:
        cols = [c for c in perm_imp.index if c.startswith(prefix)]
        perm_imp_collapsed[prefix.rstrip("_")] = perm_imp_collapsed[cols].sum()
        perm_imp_collapsed = perm_imp_collapsed.drop(cols)

    return perm_imp_collapsed.sort_values()

def get_perm_importance(X_train, y_train, X_test, y_test, model, dummy_prefixes = []):
    result_train, result_test = perm_imp(X_train, y_train, X_test, y_test, model)
    perm_imp_train_collapsed = perm_imp_collapse(result_train, X_train, dummy_prefixes)
    perm_imp_test_collapsed = perm_imp_collapse(result_test, X_test, dummy_prefixes)
    return perm_imp_train_collapsed, perm_imp_test_collapsed