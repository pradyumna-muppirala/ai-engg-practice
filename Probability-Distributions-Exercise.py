import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, uniform, skew, kurtosis
import seaborn as sns
import pandas as pd

# Gaussian distribution
x= np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, loc=0, scale=1), color="blue", label="Gaussian (mue = 0, s=1)")

#Binomial distribution
n, p = 10, 0.5
x = np.arange(0, n+1)
plt.bar(x, binom.pmf(x,n,p), color="red", label="Binomial(n=10, p=0.5)")

#Poisson distribution
lam = 3
x = np.arange(0, 10)
plt.bar(x, poisson.pmf(x, lam), color="green", label="Poisson (l = 3)")


#Uniform distribution
x = np.random.uniform(low=0, high=10, size=1000)
sns.histplot(x, kde=True, label="Uniform", color="orange")

plt.legend()
plt.show()
# Hands-on exercises

sales_df = pd.read_csv("sales_data.csv")
Sales_Amounts = sales_df["Sales_Amount"].to_list()
Units_Sold = sales_df["Units_Sold"].to_list()

# Analyse Units sold
print("Skewness : ", skew(Units_Sold))
print("Kurtosis ", kurtosis(Units_Sold))
sns.histplot(Units_Sold, kde=True)
plt.title("Distribution of Units Sold")
plt.show()