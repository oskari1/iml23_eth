import pandas as pd
test_data = pd.read_csv("./task0_sl19d1/test.csv")

def row_mean(x):
    return x[1:].mean()

output_series = test_data.apply(row_mean, axis = 1)
output_frame = output_series.to_frame()
output_frame.columns = ['y']
output_frame.index = pd.Series((i for i in range(10000, 10000 + len(output_series))))
output_frame = output_frame.rename_axis('Id')
#print(output_frame.head())
output_frame.to_csv('output.csv')
