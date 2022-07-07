import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/Melbourne Housing Snapshot/'

df = pd.read_csv(directory+'melb_data.csv')

# Drop rows with NaN value.
df = df.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

X = df[features]
y = df['Price']
# print(X.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model_rfr = RandomForestRegressor(random_state=1)
model_rfr.fit(X_train, y_train)
y_pred = model_rfr.predict(X_test)

print(mean_absolute_error(y_test, y_pred))
