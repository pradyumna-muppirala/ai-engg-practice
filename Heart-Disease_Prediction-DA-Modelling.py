import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency


hd_df = pd.read_csv("Heart_Disease_Prediction.csv")

#Inspect the data
print(hd_df.info())
print(hd_df.describe())

# Visualize the distributions
sns.histplot(hd_df["BP"], kde=True)
plt.title("Distribution of Blood Pressure Values in Give data set")
plt.show()

# Map 'Yes' to 1 and 'No' to 0 to 'Presence' and 'Absence' of Heart Disease
hd_df['Heart Disease'] = hd_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
sns.heatmap(hd_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heat map")
plt.show()

# Hypthesis testing 
presence_bp = hd_df[hd_df['Heart Disease'] == 1]['BP']
absence_bp = hd_df[hd_df['Heart Disease'] == 0]['BP']

# Perform t-test
t_stat, p_value = ttest_ind(presence_bp, absence_bp)
print("T-Stat", t_stat, "P-value", p_value)

#Interpret results
alpha = 0.05
if (p_value <= alpha):
    print("Reject all Null hypothesis - there is significant difference or effect")
else:
    print("Failed to reject Null hypothesis - No signficant difference or effect")

#Linear regression model
x = np.array(hd_df['Age']).reshape(-1,1)
y = np.array(hd_df['Cholesterol'])
z = np.array(hd_df['BP'])


#Fit linear regression
model = LinearRegression()
model.fit(x, y)

model2 = LinearRegression()
model2.fit(x, z)


print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
print("R-squared: ", model.score(x, y))

plt.scatter(x, y, color="blue", label="Cholesteorol")
plt.scatter(x, z, color="green", label="Blood Pressure")
plt.plot(x, model.predict(x), color="red", label="Cholesteorol")
plt.plot(x, model2.predict(x), color="orange", label="Blood Pressure")
plt.xlabel("Age")
plt.ylabel("Cholesteorol")
plt.legend()
plt.title("Linear Regression - Age vs Cholesteorol vs Blood Pressure")
plt.show()

contingency_table = pd.crosstab(hd_df['Age'], hd_df['BP'])
#Perform chi-squared test
chi2, p_value , dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic :", chi2)
print("P-value", p_value)
print("dof", dof)
print("Expected frequencies :\n", expected) 
alpha = 0.05

if (p_value <= alpha):
    print("Reject null hypothesis - variables are dependent")
else:
    print("Failed to reject Null hypothesis - variables are independent")
