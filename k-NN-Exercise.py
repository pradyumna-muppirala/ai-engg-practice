import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,  Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

hd_df = pd.read_csv("Heart_Disease_Prediction.csv")
# Linear regression
features = hd_df[['Age', 'BP', 'Cholesterol' , 'Max HR']]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

target=hd_df['Heart Disease']

print("Features \n", features.head())
print("Target \n", target.head())
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

#Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Experiment with different values for K

k = 18
#Initialize the kNN model
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, y_train)

#Predict on test data
y_pred = knn.predict(X_test)

#Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("k = {k} => KNN Accuracy score : ", accuracy)

lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict using Logistic regression
y_pred_lr = lr.predict(X_test)

#Evaluate the performance of Logistic Regression
accuracy = accuracy_score(y_pred_lr, y_test)
print("k = {k} => LR Accuracy score : ", accuracy)




