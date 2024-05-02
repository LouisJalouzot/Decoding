import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import memory

alphas = np.logspace(-5, 8, 10)


# @memory.cache
def train_ridge(X, Y, alphas=alphas):

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, alpha_per_target=True)),
        ]
    )
    pipeline = TransformedTargetRegressor(
        pipeline, transformer=StandardScaler(), check_inverse=False
    )
    pipeline.fit(X, Y)

    return pipeline
