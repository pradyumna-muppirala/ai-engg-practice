import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sales_df = pd.read_csv("sales_data.csv")
Sales_Amounts = sales_df["Sales_Amount"].to_list()
Units_Sold = sales_df["Units_Sold"].to_list()

#Utility function
def transpose_column_list(row_list):
    # Transpose into row-type list
    return (np.array(row_list).reshape(1,len(row_list)))
 
#Task 1 : implement mathematcial formula for linear regression
def predict (X, theta):
    return np.dot(X, theta)

#Task 2: Use gradient descent to optimize the model parameters
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)

    for _ in range(iterations):
        try:
            gradients = 1/m * np.dot(transpose_column_list(X),(predict(X, theta)-y))
            theta -= learning_rate * gradients
        except (RuntimeWarning):
            print("Catching overflow error to see the internal state")
        
    return theta

#Task 3: Calculate Evaluation Metrics
def mean_square_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true)**2))
    return (1-(ss_res/ss_tot))


X = Units_Sold
y = Sales_Amounts

X_b = np.c_[np.ones((len(Units_Sold),1)), X]

theta = np.random.randn(1000,1)
learning_rate = 0.1
iterations = 1000

# Perform gradient descent
print(theta)
theta_optimized  = gradient_descent(X, y, theta, learning_rate, iterations)

#Predictions and Evaluations
y_pred = predict(X, theta_optimized)
mse = mean_square_error(y_pred, y)
r2 = r_squared(y_pred, y)

print(y_pred)

print("Optimized Theta ", theta_optimized)
print(" Mean Squared Error ", mse)
print(" R squared ", r2)

