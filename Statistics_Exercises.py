import numpy as np
from statistics import mode
from scipy.stats import stats
from scipy.stats import ttest_ind
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