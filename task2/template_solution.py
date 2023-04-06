# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

input_scaler = 0
output_scaler = 0

def create_time_line_column(X):
    _, rows = X.shape
    X[:,0] = np.linspace(1, rows, rows)
    return X

def encode_season_column(X):
    """ encode seasons as {winter -> 0, spring -> 1, summer -> 2, autumn -> 3} """
    season_col = X[:,0]
    encode_season = lambda season: ( 
        0 if season == "winter" else (
        1 if season == "spring" else (
        2 if season == "summer" else (
        3
        ))))
    encoded_seasons = np.array(list(map(encode_season, season_col)))
    X[:,0] = encoded_seasons
    return X

def data_imputation(df):
    """ 
    Perform data imputation by taking average between last and next non-null value.
    
    Parameters
    ----------
    df: pd.DataFrame where first column is the season-column with strings and the other ones contain prices (floats) per season
    ----------
    Returns
    ----------
    df: pd.DataFrame where each null-value is replaced by the average of the previous and next non-null values of that column.
        If there is no previous non-null value in that column, it is replaced by the first non-null value of that column.
        If there there is no next non-null value in that column, it is replaced by the last non-null value of that column.
        output df has same dimensions as input df
    """
    cols = df.columns[1:].to_list()
    first_vals = list(cols) 
    last_vals = list(cols)
    # build maps which map each column to the first/last non-null value in that column
    for i, col in enumerate(df.columns[1:]):
        col_series = df[col].dropna()
        first_vals[i] = col_series.to_list()[0]
        last_vals[i] = col_series.to_list()[-1]

    first_val_of_col = dict(zip(cols, first_vals))
    last_val_of_col = dict(zip(cols, last_vals))

    # iterate through each column filling in missing values by the average of the previous and next non-null value 
    for col in df.columns[1:]:
        is_nan = df[col].isna() 
        # forward-pass by inserting previous non-null value into NaN cells (needed to take average in backward-pass)
        prev_val = first_val_of_col[col]
        for i, _ in enumerate(df[col]):
            if(is_nan[i]):
                df.at[i, col] = prev_val
            else:
                prev_val = df.at[i, col]  
        # backward-pass by finding next non-null value and taking average of last and next non-null value
        len = df[col].size 
        next_val = last_val_of_col[col]
        for i, _ in enumerate(df[col]):
            j = len - 1 - i
            if(is_nan[j]):
                df.at[j, col] = np.average([df.at[j,col], next_val]) 
            else:
                next_val = df.at[j,col]

    #plt.figure(figsize=(20,6))
    #column = 'price_CHF'
    #sns.lineplot(data=df[column], color='r', label="imputed data")
    #sns.lineplot(data=df_old[column], color='b', label="original data")
    #sns.scatterplot(x=df_old.index, y=df_old['price_POL'], color='b')
    #sns.scatterplot(x=df.index, y=df['price_POL'], color='r')
    #plt.show()

    # Check that all NaN entries are removed 
    assert df.dropna().shape == df.shape
    return df
 

def data_loading():
    """
    This function loads the training and test data, replaces the NaNs by interpolated values (see data_imputation),
    encodes the seasons to integers and normalizes the data (recommended for Gaussian Process).

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(3))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(3))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    # perform data imputation 
    old_train_df_shape = train_df.shape
    old_test_df_shape = test_df.shape

    train_df = data_imputation(train_df)
    test_df = data_imputation(test_df)

    # plot price_CHF vs price_GER 
    #sns.scatterplot(x=df.index, y=df['price_POL'], color='r')
    #sns.scatterplot(x=train_df['price_GER'], y=train_df['price_CHF'], color='b')
    #plt.show()
    # normalize the data first
    #train_arr = train_df[["price_CHF", "price_GER"]].to_numpy()
    #scaler = preprocessing.StandardScaler().fit(train_arr)
    #train_arr_scaled = scaler.transform(train_arr)
    #train_df_scaled = pd.DataFrame({'price_GER':train_arr_scaled[:,1], 'price_CHF':train_arr_scaled[:,0]})
    #sns.scatterplot(x=train_df_scaled['price_GER'].head(100), y=train_df_scaled['price_CHF'].head(100), color='b')
    #sns.scatterplot(x=train_df_scaled['price_GER'], y=train_df_scaled['price_CHF'], color='b')
    #plt.show()

    old_X_train_shape = X_train.shape
    old_y_train_shape = y_train.shape
    old_X_test_shape = X_test.shape

    # extract X_train, y_train
    X_train = train_df.drop(['price_CHF'], axis=1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()

    # encode season-column
    X_train = encode_season_column(X_train)
    X_test = encode_season_column(X_test)

    # normalize the data (recommended for Gaussian processes)
    global input_scaler 
    input_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = input_scaler.transform(X_train)
    X_test = input_scaler.transform(X_test)
    rows, _ = X_train.shape
    y_train_reshaped = y_train.reshape((rows, 1))
    global output_scaler 
    output_scaler = preprocessing.StandardScaler().fit(y_train_reshaped) 
    y_train = output_scaler.transform(y_train_reshaped).reshape(old_y_train_shape)

    # sanity check of dimensions
    assert (old_X_test_shape == X_test.shape) and (old_X_train_shape == X_train.shape) and (old_y_train_shape == y_train.shape)
    assert (old_train_df_shape == train_df.shape) and (old_test_df_shape == test_df.shape)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    ls_list = [1e-4, 1e-3, 1e-2, 1e-1]
    k = 10 
    mean_scores = list(ls_list)

    for i, ls in enumerate(ls_list):
        print("Using length_scale = {}".format(ls))
        #ls = np.full((10,), ls)
        kernel = RBF(length_scale=ls)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0) 
        scores = cross_val_score(gpr, X_train, y_train, cv=k)
        mean_scores[i] = scores.mean()
        print("Got mean score of {} for length_scale = {}".format(mean_scores[i], ls))

    best_ls = ls_list[mean_scores.index(max(mean_scores))]
    print("best length_scale = {}".format(best_ls))
    #best_ls_arr = np.full((10,), best_ls)
    gpr = GaussianProcessRegressor(kernel=RBF(length_scale=best_ls), random_state=0).fit(X_train, y_train)
    y_pred_scaled = gpr.predict(X_test)
    # need to rescale since the Gaussian regressor is dealing with standardized data (zero mean, unit variance)
    rows, _ = X_test.shape
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape((rows,1))) 
    y_pred = y_pred.reshape(y_pred_scaled.shape)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

