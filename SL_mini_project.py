#Supervised Learning mini-project

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
from sklearn.preprocessing import PolynomialFeatures

hd_df = pd.read_csv("Heart_Disease_Prediction.csv")

print(hd_df.info())
print(hd_df.describe())

#visualize relationships
sns.pairplot(hd_df, )
# Linear regression
features = hd_df[['Age', 'BP', 'Cholesterol' , 'Max HR']]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

target=hd_df['Heart Disease']

# print("Features \n", features.head())
# print("Target \n", target.head())


# print(hd_df.info())
# print(hd_df.describe())

#visualize relationships
# sns.pairplot(hd_df, vars=['Age', 'BP', 'Cholesterol' , 'Max HR'])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

#Classification by Logistic and kNN classification algorithms
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

poly_features = PolynomialFeatures(degree=2,  include_bias=False)
X_poly = poly_features.fit_transform(X_train)


print("Training data set", X_train.shape)
print("Testing data set", X_test.shape)
print("Y Training data set", y_train.shape)
print("Y_testing data set", y_test.shape)

#Fit Linear regression
model = Lasso(alpha=1)
model.fit(X_poly, y_train)

# Make predictions
X_test_poly = poly_features.fit_transform(X_test)
y_pred_lasso = model.predict(X_test_poly)

#Print Co-efficients
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
mse=mean_squared_error(y_test, y_pred_lasso)
r2 = r2_score(y_test, y_pred_lasso)

#Print Co-efficients
print("Lasso MSE : ", mse)
print("R squared: ", r2)
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
plt.scatter(X_test, y_pred_lasso, color="blue", label="BP vs Max HR")
plt.scatter(X_train, y_train, color="green", label="Training Data")
plt.plot(X_test, y_pred, color="red", label="Max HR")
plt.xlabel("BP")
plt.ylabel("Max HR")
plt.legend()
plt.title("Lasso model - Linear Regression - Blood Pressure vs Max HR")
plt.show()

#Integration graph

plt.scatter(X_test, y_test, color="blue", label="BP vs Max HR")
plt.scatter(X_train, y_train, color="green", label="Training Data")
plt.plot(X_test, y_pred_linear, color="red", label="Max HR")
plt.plot(X_test, y_pred_lasso, color="yellow", label="Max HR")
plt.plot(X_test, y_pred_ridge, color="orange", label="Max HR")
plt.xlabel("BP")
plt.ylabel("Max HR")
plt.legend()
plt.title("Linear vs Lasso vs Ridge model - Linear Regression - Blood Pressure vs Max HR")
plt.show()
