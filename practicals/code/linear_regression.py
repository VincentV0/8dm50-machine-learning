import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta


def fit_to_model(beta, X):
    """
    Parameters
    ----------
    beta : TYPE np.array
        Contains the linear model coefficients.
    X : TYPE np.array
        Contains dataset features (column with ones not included).

    Returns
    -------
    None.

    """
    
    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    
    # apply the coefficients to the data
    y = np.dot(X, beta)
    return y


def mean_squared_error(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : TYPE np.array
        Contains the true targets.
    y_pred : TYPE np.array
        Contains the predicted targets.

    Returns
    -------
    None.

    """
    
    # check if shapes of the arrays match
    if y_true.shape != y_pred.shape:
        print('y_true and y_pred should have the same shape!')
    
    # calculate MSE
    MSE = np.sum((y_true-y_pred)**2);
    MSE /= len(y_true)
    return MSE
    

def wlsq(X, y, d):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :param d: Weight vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # normalize weights
    d = (1/np.max(d))*np.array(d)


    # calculate the coefficients
    diagDXtX = np.dot(np.diag(d), (np.dot(X.T, X)))
    beta = np.dot(np.linalg.inv(diagDXtX), np.dot(X.T, y))

    return beta