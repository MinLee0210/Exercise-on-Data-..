import pandas as pd
import numpy as np 

directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/'
df = pd.read_csv(directory + 'airquality.csv')

df = df.dropna(axis=0).reset_index(drop=True)
df = df.drop(['Day'], axis=1)

# options = [5]
# print(df[df['Month'].isin(options)].Temp)

df_title = df.columns
df_month = set(df['Month'])

for i in df_month:
    print(np.average(df[df['Month'] == i].Temp))