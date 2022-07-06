from random import Random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/Housing_Price/'

x_full = pd.read_csv(directory + 'train.csv', index_col='Id')
x_test_full = pd.read_csv(directory + 'test.csv', index_col='Id')

y = x_full['SalePrice']
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = x_full[features].copy()
x_test = x_test_full[features].copy()

X_train, X_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error',random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

# for i in range(0, len(models)):
#     mae = score_model(models[i])
#     print("Model %d MAE: %d" % (i + 1, mae))

my_model = RandomForestRegressor(n_estimators=100, random_state=0)
my_model.fit(x, y)
ypred_rfr = my_model.predict(x_test)

output = pd.DataFrame({'Id': x_test.index, 
                       'SalePrice': ypred_rfr})
output.to_csv('Submission_exercise_3', index=False)