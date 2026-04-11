from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def get_gbr(random_state=42):
    """
    Gradient Boosting Regressor for approximating dispersion curves:
    input  -> (kL, L)
    output -> c / beta_l
    """
    return GradientBoostingRegressor(
        n_estimators=300,        # boosting iterations
        learning_rate=0.05,      # step size (prevents overfitting)
        max_depth=4,             # controls model complexity
        subsample=0.9,           # improves generalization
        random_state=random_state
    )


def get_rf(random_state=42):
    """
    Random Forest Regressor as a baseline model for comparison
    """
    return RandomForestRegressor(
        n_estimators=300,        # number of trees
        max_depth=None,          # allow full growth
        n_jobs=-1,               # parallel computation
        random_state=random_state
    )