import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier


directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/Housing_Price/'

x_full = pd.read_csv(directory + 'train.csv', index_col='Id')
x_test_full = pd.read_csv(directory + 'test.csv', index_col='Id')

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = x_full[features]
y = x_full['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model_dtc = DecisionTreeClassifier(random_state=1)
model_dtc.fit(X_train, y_train)
y_pred = model_dtc.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(mae))