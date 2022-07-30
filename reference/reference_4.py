import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/Housing_Price/'
hp_data = pd.read_csv(directory+'train.csv', index_col='Id')

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
y = hp_data['SalePrice']
X = hp_data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model_rfr = RandomForestRegressor(random_state=1)
model_rfr.fit(X_train, y_train)
y_pred = model_rfr.predict(X_test)
mae_val = mean_absolute_error(y_pred, y_test)

print("Validation MAE for Random Forest Model: {:,.0f}".format(mae_val))