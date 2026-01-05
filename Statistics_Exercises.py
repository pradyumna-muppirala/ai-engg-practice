import numpy as np
from statistics import mode
from scipy.stats import stats
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import ttest_1samp
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

data = [10, 20, 30, 40, 50]
mean = sum(data)/len(data)
print("Mean: ", mean)

sorted_data = sorted(data)
median = sorted_data[len(data)//2] if (len(data) % 2 != 0) else \
    (0.5*(sorted_data[len(data)//2]+sorted_data[len(data)//2-1]))
print("Median : ", median)

print("Mode: ", mode(data))

variance = sum((x-mean) ** 2 for x in data)/ len(data)
print("Variance =>", variance)

std = np.sqrt(variance)
print("Standard Dev => ", std)

#Hypothesis 
#null Hypothesis - no effect or difference exists
#alternative Hypothesis - some effect or difference exists
z_score = 1.96
sample_mean  = mean
ci = (float(sample_mean - z_score * (std / (len(data)**0.5))), 
      float(sample_mean + z_score * (std / (len(data)**0.5))))
print("95% CI level =>" , ci)

#T-test sample exercise
group1 = [2.1, 2.5, 2.8, 3.0, 3.2]
group2 = [1320.1, 122.5, 13.8, 39.05, 20.2]

t_stat, p_value = ttest_ind(group1, group2)
print("T-statistic =>", t_stat)
print("P-value =>", p_value)

alpha = 0.05

if (p_value >= alpha):
    print("Failed to reject null hypothesis => No significant change is found in the given data sets")
else:
    print("Reject Null hypothese - Significant change is found in the given data sets")

sales_df = pd.read_csv("sales_data.csv")
Sales_Amounts = sales_df["Sales_Amount"].to_list()
Units_Sold = sales_df["Units_Sold"].to_list()

t_stat, p_value = ttest_ind(Units_Sold, Sales_Amounts)
print("T-statistic =>", t_stat)
print("P-value =>", p_value)

alpha = 0.05

if (p_value >= alpha):
    print("Failed to reject null hypothesis => No significant difference between Units_Sold vs Sales_Amounts")
else:
    print("Reject Null hypothesis - significant difference between Units_Sold vs Sales_Amounts")

z_score = 1.96
sample_mean  = np.mean(Sales_Amounts)
sample_std = np.std(Sales_Amounts)
ci = (float(sample_mean - z_score * (std / (len(Sales_Amounts)**0.5))), 
      float(sample_mean + z_score * (std / (len(Sales_Amounts)**0.5))))
print("Sales Amounts => Mean, Standard deviation:" , sample_mean, sample_std)
print("95% CI level of Sales Amounts=>" , ci)

# One sample t-test exercise
data = [12, 14,  15, 16, 17, 18]
population_mean = 15

#Calculate t_stat and p_value
t_stat, p_value = ttest_1samp(data, population_mean)
print("T-Statistic : ", t_stat)
print("P-value : ", p_value)
print("Mean: ", np.mean(data))
alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis Significant difference or effect")
else:
    print("Failed to reject the null hypothesis: No significant difference")

#Two sample t-test

#Perform t-test
t_stat, p_value = ttest_ind(Sales_Amounts, Units_Sold)
print("T-Statistic : ", t_stat)
print("P-value : ", p_value)
print("Sales Amount Mean : ", np.mean(Sales_Amounts))
print("Units Sold Mean : ", np.mean(Units_Sold))
alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis Significant difference or effect")
else:
    print("Failed to reject the null hypothesis: No significant difference")

data = [[50, 30],  [20,40]]
# Perform chi-square test
chi2, p_value, dof , expected = chi2_contingency(data)
print("Chi-Square Statistic :", chi2)
print("P-value", p_value)
print("dof", dof)
print("Expected frequencies :\n", expected)  

# Anova 
group1 = [12, 14, 15, 16, 17]
group2 = [11, 13, 14, 15, 16]
group3 = [10, 12, 14, 14, 16]

f_stat, p_value = f_oneway(group1, group2, group3)
print("F-stat : ", f_stat)
print("p-value : ", p_value)

# Hands-on exercises
#Perform 1-sample T-test
population_mean = 15
t_stat, p_value = ttest_1samp(group1, population_mean)
print("1 Sample T-Test : T-stat : ", t_stat, "P-value", p_value)
#Perform 2-sample T-test
t_stat, p_value = ttest_ind(group1 , group2)
print("2 sample T-Test : T-stat : ", t_stat, "P-value", p_value)
#Perform paired sample T-test
t_stat, p_value = ttest_rel(group1 , group3)
print("Paired T-Test : T-stat : ", t_stat, "P-value", p_value)

#Perform chi-squared test
list1 = []
list1.append(group1)
list1.append(group3)
print(list1)
chi2, p_value, dof , expected = chi2_contingency(list1)
print("Chi-Square Statistic :", chi2)
print("P-value", p_value)
print("dof", dof)
print("Expected frequencies :\n", expected)  

#ANOVA Hands-on practice
group4 = group1 * 4
group5 = group2 * 3
group6 = group3 * 9

f_stat, p_value = f_oneway(group4, group5, group6)
print("F-stat : ", f_stat)
print("p-value : ", p_value)

list1.clear()
list1.append(Sales_Amounts)
list1.append(Units_Sold)

print(list1)
chi2, p_value, dof , expected = chi2_contingency(list1)
print("Chi-Square Statistic :", chi2)
print("P-value", p_value)
print("dof", dof)
print("Expected frequencies :\n", expected)  

#Pearson correlation
r, _ = pearsonr(Units_Sold, Sales_Amounts)
print("Pearson Correlation Coeffcient", r)

# Spearman correlation
rho, _ = spearmanr(Units_Sold, Sales_Amounts)
print("Spearman Correlation Coefficient", rho)

#Linear Regression 
x = np.array([1 , 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 8, 10])

#Fit linear regression
model = LinearRegression()
# x = np.array(Units_Sold).reshape(-1, 1) 
# y = Sales_Amounts
model.fit(x, y)

print("Slope : ", model.coef_[0])
print("Intercept : ", model.intercept_)
print("R-squared: ", model.score(x, y))

plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, model.predict(x), color="red", label="Regression Line")
plt.legend()
plt.title("Linear Regression - Units Sold vs Sales Amounts")
plt.show()

#Correlation matrix
del sales_df["Drug_ID"]
del sales_df["Drug_Name"]
del sales_df["Category"]
del sales_df["Country"]
del sales_df["Date"]

correlation_matrix = sales_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


