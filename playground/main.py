import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("train.csv")
df_old = pd.read_csv("train.csv")
#print(df)
#print("\n")
# extract columns starting from price_AUS
cols = df.columns[1:].to_list()
first_vals = list(cols) 
# build map which maps each column to the first non-null value in that column
for i, col in enumerate(df.columns[1:]):
    col_series = df[col].dropna()
    first_vals[i] = col_series.to_list()[0]
first_val_of_col = dict(zip(cols, first_vals))

# iterate through each column filling in missing values by the last non-null value 
for col in df.columns[1:]:
    prev_val = first_val_of_col[col]
    is_nan = df[col].isna() 
    for i, val in enumerate(df[col]):
        if(is_nan[i]):
            df.at[i, col] = prev_val
        else:
            prev_val = df.at[i, col]  

print(df)
print("\n")
print(df_old)
plt.figure(figsize=(20,6))
column = 'price_POL'
sns.lineplot(data=df[column], color='r', label="imputed data")
sns.lineplot(data=df_old[column], color='b', label="original data")
#sns.scatterplot(x=df_old.index, y=df_old['price_POL'], color='b')
#sns.scatterplot(x=df.index, y=df['price_POL'], color='r')
plt.show()