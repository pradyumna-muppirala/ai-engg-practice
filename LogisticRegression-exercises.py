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

#Sigmoid function
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

#Generate values 
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.plot(z, sigmoid_values)
plt.title("Sigmoid function")
plt.xlabel("z")
plt.ylabel("sigma(z)")
plt.grid()
plt.show()


hd_df = pd.read_csv("Heart_Disease_Prediction.csv")
# Logistic regression
features = hd_df[["Age", "Max HR"]]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

target=hd_df['BP']

print("Features \n", features.head())
print("Target \n", target.head())

#Train , Test and Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
print("Training data set", X_train.shape)
print("Testing data set", X_test.shape)

#Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#Evaluate performance
print("Accurancy :", accuracy_score(y_test, y_pred))
print("Precision :", accuracy_score(y_test, y_pred))
print("Recall score :", accuracy_score(y_test, y_pred))
print("f1 score :", accuracy_score(y_test, y_pred))
print("classification report : \n", classification_report(y_test, y_pred))

# Plot decision boundary
x_min, x_max = features["Age"].min() - 10, features["Age"].max() + 10
y_min = features["Max HR"].min() -30
y_max = features["Max HR"].max() + 30

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# Predict probabilities for grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#Plot
print("X test shape : ", X_test.shape)
print("Y test shape : ", y_test.shape)
print("Y pred shape :", y_pred.shape)
print("X test head :\n", X_test.head())
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
# plt.scatter(X_test["Age"], X_test["Max HR"], c=y_test,  edgecolor="k", cmap="coolwarm")
plt.scatter(X_test["Age"], X_test["Max HR"], c=y_pred,  edgecolor="k", cmap="coolwarm")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Max HR")
plt.show()

# plt.scatter(X_test, y_test, color="blue", label="BP vs Max HR")
# plt.scatter(X_train, y_train, color="green", label="Training Data")
# plt.plot(X_test, y_pred, color="red", label="Max HR")
# plt.xlabel("BP")
# plt.ylabel("Max HR")
# plt.legend()
# plt.title("Linear Regression - Blood Pressure vs Max HR")
# plt.show()

model = RandomForestClassifier(random_state=42)
kf=KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores=cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

#Output results
print("Cross validation scores:", cv_scores)
print("Mean accurancy:", cv_scores.mean())

model=LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

#Display confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels="BP")
disp.plot(cmap="Blues")
plt.show()

#Print report Classification report
print("Classification report:\n", classification_report(y_test, y_pred))
