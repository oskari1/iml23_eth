# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# ADDED
#from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import cross_val_score

def transform_row(x):
    """
    Parameters
    ----------
    x: array of floats, dim = (1,5)

    Returns
    ----------
    x_transformed: array of floats, dim = (1, 21)
    """
    x_transformed = np.zeros((1, 21))
    x_transformed[0,0:5] = x
    x_transformed[0,5:10] = x*x
    x_transformed[0,10:15] = np.exp(x)
    x_transformed[0,15:20] = np.cos(x)
    x_transformed[0,20] = 1
    return x_transformed
# END

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here
    X_transformed = (np.apply_along_axis(transform_row, 1, X)).reshape((700,21))
    # END
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here

    # chosen parameters
    lambdas = [i/10 for i in range (1,1000)] # best so far range (1, 1000)
    k = 10 # best so far 10

    mean_scores = list(lambdas) 
    max_iter = 1000 # best so far 10000
    epsilon = 1
    for i, lam in enumerate(lambdas):
        model = HuberRegressor(alpha=lam, fit_intercept=False, max_iter = max_iter, epsilon=epsilon) # maybe use fit_intercept = False, maybe use Lasso
        scores = cross_val_score(model, X_transformed, y, scoring="neg_mean_squared_error", cv=k)
        mean_scores[i] = scores.mean()

    #best_lam = lambdas[mean_scores.index(min(mean_scores))]
    best_lam = lambdas[mean_scores.index(max(mean_scores))]
    best_model = HuberRegressor(alpha=best_lam, fit_intercept=False, max_iter = max_iter, epsilon=epsilon) # maybe use fit_intercept = False, maybe use Lasso
    w = best_model.fit(X_transformed, y).coef_

    # plot lambda vs mean_score
    print(best_lam) 
    plt.plot(lambdas, mean_scores)
    plt.xlabel("lambda")
    plt.ylabel("mean_cross_val_score")
    plt.show()

    # END
    assert w.shape == (21,)
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
