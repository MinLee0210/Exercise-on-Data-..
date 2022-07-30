# This exercise uses the dataset of abalone to do 2 task: 
#     Use Logistic Regression and chose the subset with part of dataset features to predict its Gender. 
#     Do the same as above but using Softmax instead. 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder


directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/Abalone/'
titles = ['Sex', 'Length', 'Height', 'Diameter', 'Whole weight', 'Shucked weight', 
          'Viscera weight', 'Shell weight', 'Rings']

df = pd.read_csv(directory + 'abalone.data')
df.columns = titles

X = df[['Length', 'Height', 'Diameter', 'Whole weight']]
y = df['Sex']

# Encode Y from categorical data into numerical data
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.5 ,random_state=0)

model_lr = LogisticRegression(solver= 'liblinear')
model_lr.fit(X_train, y_train)
y_pred_1 = model_lr.predict(X_test)

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_2 = model_linear.predict(X_test)

model_knn = KNeighborsClassifier(n_neighbors= 3).fit(X_train, y_train)
y_pred_3 = model_knn.predict(X_test)

model_rfr = RandomForestRegressor(random_state=1)
model_rfr.fit(X_train, y_train)
y_pred_4 = model_rfr.predict(X_test)

acc1 = accuracy_score(y_test, y_pred_1)
# acc2 = accuracy_score(y_test, y_pred_2) :-> The problem occurs because expect the input value to be not continous value. 
#                                             In other words, it accepts value which, is either Integer or Label Encoder. 
acc3 = accuracy_score(y_test, y_pred_3)

mse = mean_squared_error(y_test, y_pred_2)
mse_2 = mean_squared_error(y_test, y_pred_4)


print("Accuracy score of using Logistic Regression: {}".format(acc1))
print("Accuracy score of using KNeighborClassifier: {}".format(acc3))

print("Mean Squared Error while using Linear Regression: {}".format(mse))
print("Mean Squared Error of using Random Forest: {}".format(mse_2))