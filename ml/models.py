from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def get_gbr():
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4
    )

def get_rf():
    return RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1
    )
