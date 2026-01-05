import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,  Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


hd_df = pd.read_csv("Heart_Disease_Prediction.csv")
# Linear regression
features = hd_df[['BP']]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

target=hd_df['Max HR']

print("Features \n", features.head())
print("Target \n", target.head())

#Train , Test and Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
print("Training data set", X_train.shape)
print("Testing data set", X_test.shape)

#Fit Linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print Co-efficients
print("MSE : ", mse)
print("R squared: ", r2)
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
plt.scatter(X_test, y_test, color="blue", label="BP vs Max HR")
plt.scatter(X_train, y_train, color="green", label="Training Data")
plt.plot(X_test, y_pred, color="red", label="Max HR")
plt.xlabel("BP")
plt.ylabel("Max HR")
plt.legend()
plt.title("Linear Regression - Blood Pressure vs Max HR")
plt.show()

#Polynominal regressions
#Features
features = hd_df[['BP']]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
target=hd_df['Max HR']

#Train , Test and Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

poly_features = PolynomialFeatures(degree=2,  include_bias=False)
X_poly = poly_features.fit_transform(X_train)

print("Features \n", features.head())
print("Target \n", target.head())



print("Training data set", X_train.shape)
print("Testing data set", X_test.shape)
print("Y Training data set", y_train.shape)
print("Y_testing data set", y_test.shape)

#Fit Linear regression
model = LinearRegression()
model.fit(X_poly, y_train)

# Make predictions
X_test_poly = poly_features.fit_transform(X_test)
y_pred_linear= model.predict(X_test_poly)

#Print Co-efficients
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
mse=mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

#Print Co-efficients
print("MSE : ", mse)
print("R squared: ", r2)
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
plt.scatter(X_test, y_test, color="blue", label="BP vs Max HR")
plt.scatter(X_train, y_train, color="green", label="Training Data")
plt.plot(X_test, y_pred_linear, color="red", label="Max HR")
plt.xlabel("BP")
plt.ylabel("Max HR")
plt.legend()
plt.title("Polynomial - Linear Regression - Blood Pressure vs Max HR")
plt.show()

# Regularization techniques
#Ridge 

#Features
features = hd_df[['BP']]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
target=hd_df['Max HR']

#Train , Test and Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

poly_features = PolynomialFeatures(degree=2,  include_bias=False)
X_poly = poly_features.fit_transform(X_train)

print("Features \n", features.head())
print("Target \n", target.head())



print("Training data set", X_train.shape)
print("Testing data set", X_test.shape)
print("Y Training data set", y_train.shape)
print("Y_testing data set", y_test.shape)

#Fit Linear regression
model = Ridge(alpha=1)
model.fit(X_poly, y_train)

# Make predictions
X_test_poly = poly_features.fit_transform(X_test)
y_pred_ridge = model.predict(X_test_poly)

#Print Co-efficients
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
mse=mean_squared_error(y_test, y_pred_ridge)
r2 = r2_score(y_test, y_pred_ridge)

#Print Co-efficients
print("Ridge MSE : ", mse)
print("R squared: ", r2)
print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
plt.scatter(X_test, y_test, color="blue", label="BP vs Max HR")
plt.scatter(X_train, y_train, color="green", label="Training Data")
plt.plot(X_test, y_pred_ridge, color="red", label="Max HR")
plt.xlabel("BP")
plt.ylabel("Max HR")
plt.legend()
plt.title("Ridge Model - Linear Regression - Blood Pressure vs Max HR")
plt.show()

#Lasso
#Features
features = hd_df[['BP']]
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
target=hd_df['Max HR']

#Train , Test and Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

poly_features = PolynomialFeatures(degree=2,  include_bias=False)
X_poly = poly_features.fit_transform(X_train)

print("Features \n", features.head())
print("Target \n", target.head())

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
