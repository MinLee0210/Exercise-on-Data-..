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

Q1 = df['Na_to_K'].quantile(0.25)
Q3 = df['Na_to_K'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5*IQR
lower = Q1 = 1.5*IQR

df = df[(df['Na_to_K'] < upper) & (df['Na_to_K'] > lower)]

sns.boxplot(df['Na_to_K'])
plt.show()