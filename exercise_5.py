import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/'
# df = pd.read_csv(directory + 'concrete.csv')
iris = load_iris()
X = iris["data"][:, 3: ]
y = (iris["target"] == 2)

LGmodel = LogisticRegression(solver='liblinear', random_state=1)
LGmodel.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = LGmodel.predict_proba(X_new)

plt.plot(X_new, y_prob[:, 1], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_prob[:, 0], 'rx', label='Not Iris-Virginica')
plt.grid()
plt.legend()
plt.show()
