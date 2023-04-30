import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

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

train_df = pd.read_csv("train.csv")
old_df = train_df

cols = train_df.columns
print(cols)
print(cols.drop('price_CHF'))
print("Training data:")
print("Shape:", train_df.shape)
print(train_df.head(3))
print('\n')

# Dummy initialization of the X_train, X_test and y_train   
X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
y_train = np.zeros_like(train_df['price_CHF'])

train_arr = train_df.to_numpy()
train_arr = encode_season_column(train_arr)
imputer = KNNImputer(n_neighbors=5)
train_imputed = imputer.fit_transform(train_arr)
X_train[:,0:2] = train_imputed[:,0:1]
X_train[:,2:] = train_imputed[:,3:]
y_train = train_imputed[:,2]

imputed_df = pd.DataFrame(data=train_imputed, columns=cols)
print(imputed_df.head(5))

column = 'price_CHF'
plt.figure(figsize=(20,6))
sns.lineplot(data=imputed_df[column], color='r', label="imputed data")
sns.lineplot(data=old_df[column], color='b', label="original data")
# sns.scatterplot(x=df_old.index, y=df_old[column], color='b')
# sns.scatterplot(x=df.index, y=df[column], color='r')
plt.show()