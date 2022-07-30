# This exercise use Abalone dataset. 

import pandas as pd 
import numpy as np 

directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/Abalone/'
titles = ['Sex', 'Length', 'Height', 'Diameter', 'Whole weight', 'Shucked weight', 
          'Viscera weight', 'Shell weight', 'Rings']

df = pd.read_csv(directory + 'abalone.data')
df.columns = titles

# 1. Identifying and handling the missing values.
# print(df.isnull().sum())
# df.dropna() -> drop any row that contains undefined values. 
# df.fillna(x) -> fill NaN parameter with the value of x. 

# 2. Encoding the categorical data.

# 3. Splitting dataset into training set and testing set. 

# 4. Feature scaling. 

##############################################################

df_outlook = df[['Sex', 'Length', 'Height', 'Diameter', 'Rings']]
print(df_outlook.head(5))
print()
print(df_outlook.groupby('Sex').describe())
print()
print(df_outlook.groupby(['Sex', 'Rings']).describe())
print()
print(df_outlook.groupby('Rings').mean())
print()
# print(df_outlook.groupby('Rings').agg(np.mean)): using agg as applied np.mean on group.
print(df_outlook.groupby('Rings').agg(lambda s: np.abs(min(s) - max(s))))
print()
print(df_outlook.groupby('Sex').groups)
print()
group_by_sex = df_outlook.groupby('Sex').groups
print(np.array((group_by_sex['F'])))
print(np.array(group_by_sex['I']))
print(np.array(group_by_sex['M']))
print()
print(df_outlook['Rings'].groupby(df_outlook['Sex']).sum())