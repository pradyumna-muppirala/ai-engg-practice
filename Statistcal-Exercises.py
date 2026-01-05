import numpy as np
from scipy.stats import norm, t
import pandas as pd

#Sample data
data = [12, 14, 15, 16, 17, 18, 19]

#Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data, ddof=1)

#95% Confidence Internval using t-distribution
n = len(data)
t_value = t.ppf(0.975, df=n-1)
margin_of_error = t_value* (std/ np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print("95% confidence interval ", ci)

sales_df = pd.read_csv("sales_data.csv")
Sales_Amounts = sales_df["Sales_Amount"].to_list()
Units_Sold = sales_df["Units_Sold"].to_list()

data = Sales_Amounts
mean = np.mean(data)
std = np.std(data, ddof=1)

#95% Confidence Internval using t-distribution
n = len(data)
z_value = norm.ppf(0.975)
t_value = t.ppf(0.975, df=n-1)
margin_of_error = t_value* (std/ np.sqrt(n))
ci = (float(mean - margin_of_error), float(mean + margin_of_error))
print("Population mean : ", mean)
print("95% confidence interval ", ci , "Z-value : ", z_value)

sample= sales_df["Sales_Amount"].sample(30)
mean = sample.mean()
std = sample.std()
n = len(sample)
z_value = norm.ppf(0.975)
margin_of_error = z_value * (std / np.sqrt(n))
ci = (float(mean-margin_of_error), float(mean+margin_of_error))

print("Sample mean :" , mean)
print("Sample set 95% CI :", ci)

