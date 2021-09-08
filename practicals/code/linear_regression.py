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
    Applies the result of the least squares linear regression to the data.
    
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
    Calculates the mean squared error from the true target values and the 
    predicted target values.
    
    Parameters
    ----------
    y_true : TYPE np.array
        Contains the true targets.
    y_pred : TYPE np.array
        Contains the predicted targets.

    Returns
    -------
    MSE : TYPE float
        Returns the mean squared error value.

    """
    
    # check if shapes of the arrays match
    if y_true.shape != y_pred.shape:
        print('y_true and y_pred should have the same shape!')
        return None
    
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
    #d = (1/np.max(d))*np.array(d)


    # calculate the coefficients
    xw = X.T * d
    beta = np.dot(np.linalg.inv(np.dot(xw, X)), np.dot(xw, y))

    return beta



def normalize(X):
    """
    Normalizes the feature array. The normalization step is done per feature.
    
    Parameters
    ----------
    X : TYPE np.array
        Contains the features of the dataset.
    
    Returns
    -------
    X : TYPE np.array
        Contains the normalized features of the dataset.
    """
    
    # loop over every feature independently
    for i in range(X.shape[1]):
        
        # for every value, subtract by the average value of that feature and
        # divide by the standard deviation.
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i]);
    
    return X



def kNN(X_train, y_train, X_test, k=5, mode='classification'):
    """
    Applies the k-Nearest Neighbor method to a dataset
    
    Parameters
    ----------
    X_train : TYPE np.array
        Contains the features of the training dataset.
    y_train : TYPE np.array
        Contains the targets of the training dataset.
    X_test : TYPE np.array
        Contains the features of the test dataset.
    k : TYPE int, optional
        The number of considered 'nearest neighbors'. The default is 5.
    mode : TYPE string, optional
        Choose between 'classification' and 'regression' mode. Determines the 
        result of this function. The default is 'classification'.

    Returns
    -------
    y_test : TYPE
        Predicted targets of the test dataset.

    """
    
    # initialize y_test array
    if mode == 'classification':
        y_test = np.zeros((X_test.shape[0], 1))
    elif mode == 'regression':
        y_test = np.zeros((X_test.shape[0], y_train.shape[1]))
    else:
        print("False parameter for 'mode'; should be either 'classification' or 'regression'.")
        return None


    for i in range(X_test.shape[0]):
        # calculate distance between test object and all train objects
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        
        # retrieve the indices of the closest neighbors
        ids_of_NN = distances.argsort()[:k]
        
        # retrieve the targets of the closest 5 neighbors
        NN_targets = y_train[ids_of_NN]

        if mode == 'classification':        
            # find the most occuring target and add this to y_test
            new_target = np.bincount(NN_targets.T[0]).argmax()
            y_test[i] = new_target
            
        elif mode == 'regression':
            # find the average of the training targets and add to y_test
            y_test[i,:] = np.mean(NN_targets, axis=0)
        
    return y_test