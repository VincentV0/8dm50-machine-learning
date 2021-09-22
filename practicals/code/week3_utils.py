import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv, ParameterGrid, train_test_split
from sklearn.utils import resample
import typing


def bootstrap_test_train_split(X: npt.ArrayLike, y: npt.ArrayLike, n_bootstrap_runs: int,
                               random_state: int = 0) -> typing.Generator[npt.ArrayLike,\
                                                                          npt.ArrayLike,\
                                                                          npt.ArrayLike,\
                                                                          npt.ArrayLike]:
    """
    A generator that samples X, y with replacement (used in bootstrapping) and splits
    into test ,train

    :param X: feature matrix
    :param y: target vector
    :param n_bootstrap_runs: amount of times the data is resampled
    :param random_state: random state of random number generators
    :yield: X and y resampled and split into train, test (X_train, X_test, y_train, y_test)
    """
    for i in range(n_bootstrap_runs):
        # Bootstrapping is sampling WITH replacement
        Xsampled, ysampled = resample(X, y, random_state=random_state, replace=True)

        yield train_test_split(Xsampled, ysampled, random_state=random_state)
