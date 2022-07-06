# UNDERSTANDING TRAIN_TEST_SPLIT

import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
with warnings.catch_warnings():
    # You should probably not use this dataset.
    warnings.filterwarnings("ignore")

x, y = load_boston(return_X_y=True)
print(x[:5])
print(y[:5])

# x = np.arange(24).reshape(12,2)
# y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4, shuffle=False, stratify=y)
# # train_test_split will divide randomly elements from the input to train set and test set
# print(x_train)
# print()
# print(x_test)
# print()
# print(y_train)
# print()
# print(y_test)

# x = np.arange(20).reshape(-1, 1)
# y = np.array(np.random.rand(20))

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 8, shuffle=False)

# model_lr = LinearRegression()
# model_lr.fit(x_train, y_train)
# print(model_lr.intercept_)
# print(model_lr.coef_)
