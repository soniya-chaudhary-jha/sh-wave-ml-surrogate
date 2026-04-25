from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def get_gbr(random_state=42):
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        random_state=random_state
    )


def get_rf(random_state=42):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )


# 🔹 NEW: Support Vector Regressor (SVM)
def get_svr():
    """
    SVR with RBF kernel (good for smooth nonlinear dispersion curves)
    Scaling is IMPORTANT → so we use pipeline
    """
    return make_pipeline(
        StandardScaler(),
        SVR(
            kernel='rbf',
            C=100,        # controls fitting strength
            gamma='scale',
            epsilon=0.01  # precision of fit
        )
    )


# 🔹 NEW: Small ANN (MLP Regressor)
def get_ann(random_state=42):
    """
    Small feedforward neural network
    Works well for function approximation
    """
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64),  # 2 layers
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=random_state
        )
    )