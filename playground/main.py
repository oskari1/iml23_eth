import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("train.csv").head(20)
df_old = pd.read_csv("train.csv").head(20)
# extract price-columns
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

X_train = df.drop(['price_CHF'], axis=1).to_numpy()
y_train = df['price_CHF'].to_numpy()

print("interpolated dataframe")
print(df)
print("original dataframe")
print(df_old)
print("X_train: ")
print(X_train)
print("y_train: ")
print(y_train)
plt.figure(figsize=(20,6))
column = 'price_CHF'
sns.lineplot(data=df[column], color='r', label="imputed data")
sns.lineplot(data=df_old[column], color='b', label="original data")
#sns.scatterplot(x=df_old.index, y=df_old['price_POL'], color='b')
#sns.scatterplot(x=df.index, y=df['price_POL'], color='r')
plt.show()