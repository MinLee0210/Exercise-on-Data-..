# SOURCE: https://www.kaggle.com/code/lakhankumawat/drugs-classification-notebook-for-beginners/notebook

from statistics import mode
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

directory = '/home/ducminh/Desktop/Academic_Study/Self-learning/Research/Dataset/'
df = pd.read_csv(directory+'drug200.csv')

cat_cols = [column for column in df.columns if df[column].dtype=='object']
num_cols = df.drop(cat_cols, axis=1).columns

# PLOTTING COUNT PLOT FOR CATEGORIES FEATURES: we can find the proportion of a certain kind of drug to an appropriate type of disease
# for column in cat_cols:
#     sns.countplot(df[column], hue=df['Drug'])
    # plt.show()
    
# PLOTTING NUMERICAL COLUMNS
# for column in num_cols:
#     plt.hist(df[column], bins=7, rwidth=0.5)
#     plt.xlabel(column)
#     plt.ylabel('count')
#     plt.show()

# CHECKING THE EXISTENCE OF OUTLIER
# for column in num_cols:
#     sns.boxplot(df[column])
#     plt.show()

# AFTER CHECKING FOR OUTLIERS, WE FOUND THAT 'Na_to_k' HAS SEVERAL OUTLIERS, 
#                               WE NOW CLEANING THE DATA BY REMOVING THE OUTLIERS

# WE REMOVE BY FINDING ITS FIRST AND THIRD QUANTILE TO FIND OUT ITS LOWER- AND UPPER- BOUND.
# THEN WE REMOVE ALL THE DATAS THAT HIGHER THAN OR LOWER THEN THE UPPERBOUND AND THE LOWERBOUND RESPECTIVELY.

# Q1 = df['Na_to_K'].quantile(0.25)
# Q3 = df['Na_to_K'].quantile(0.75)
# IQR = Q3 - Q1
# upper = Q3 + 1.5*IQR
# lower = Q1 = 1.5*IQR

# df = df[(df['Na_to_K'] < upper) & (df['Na_to_K'] > lower)]
# print(df.head(10))

# sns.boxplot(df['Na_to_K'])
# plt.show()

# CONVERT CATEGORICAL COLUMNS INTO NUMBERS USING LABEL ENCODING
for column in cat_cols:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    
# print(df.head(10))

X_train, X_test, y_train, y_test = train_test_split(df.drop('Drug', axis=1), df['Drug'], test_size=0.2, 
                                                    stratify=df['Drug'])

# WE USE DIFFENT MODEL FOR PREDICTION. (NORMALLY, WE USE 3 MODELS.)

# FIRST, WE USE LOGISTIC REGRESSION
model_lr = LogisticRegression(solver='liblinear')
model_lr.fit(X_train, y_train)
ypred_lr = model_lr.predict(X_test)
# print(confusion_matrix(y_test, ypred_lr))
# print(classification_report(y_test, ypred_lr))

# sns.heatmap(confusion_matrix(y_test, ypred_lr), annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# WE USE DECISION TREE CLASSIFIER   
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
ypred_dt = model_dt.predict(X_test)
# print(confusion_matrix(y_test, ypred_dt))
# print(classification_report(y_test,ypred_dt))

# sns.heatmap(confusion_matrix(y_test,ypred_dt),annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# WE USE RANDOM FOREST CLASSIFIER
model_rf=RandomForestClassifier()
model_rf.fit(X_train, y_train)
ypred_rf = model_rf.predict(X_test)
# print(confusion_matrix(y_test, ypred_rf))
# print(classification_report(y_test, ypred_rf))

# sns.heatmap(confusion_matrix(y_test,ypred_rf),annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# WE USE SUPPORT VECTOR MACHINE
model_svc = SVC()
model_svc.fit(X_train, y_train)
ypred_svc = model_svc.predict(X_test)
# print(confusion_matrix(y_test, ypred_svc))
# print(classification_report(y_test, ypred_svc))

# sns.heatmap(confusion_matrix(y_test,ypred_svc),annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# K-NEIGHBORS
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)
ypred_KNN = model_KNN.predict(X_test)
# print(confusion_matrix(y_test, ypred_KNN))
# print(classification_report(y_test, ypred_KNN))

# sns.heatmap(confusion_matrix(y_test,ypred_KNN),annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# NAIVE-BAYES
model_nb=MultinomialNB()
model_nb.fit(X_train,y_train)
ypred_nb=model_nb.predict(X_test)
# print(confusion_matrix(y_test,ypred_nb))
# print(classification_report(y_test, ypred_nb))

# sns.heatmap(confusion_matrix(y_test, ypred_KNN), annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()