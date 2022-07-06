import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/'
df = pd.read_csv(directory + 'airquality.csv')

df = df.dropna(axis=0).reset_index(drop=True)
df = df.drop(['Day', 'Unnamed: 0', 'Month'], axis=1)

# options = [5]
# print(df[df['Month'].isin(options)].Temp)

# df_title = df.columns
# df_month = set(df['Month'])

# for i in df_month:
#     print(np.average(df[df['Month'] == i].Temp))

x_train, x_test, y_train, y_test = train_test_split(df.drop('Temp', axis=1), df['Temp'],
                                                    test_size=0.2)
# print(x_train[:5])

model_lr = LogisticRegression(solver='liblinear')
model_lr.fit(x_train, y_train)
ypred_lr = model_lr.predict(x_test)

print(confusion_matrix(y_test, ypred_lr))
print(classification_report(y_test, ypred_lr))

sns.heatmap(confusion_matrix(y_test, ypred_lr), annot=True)
plt.grid()
plt.show()