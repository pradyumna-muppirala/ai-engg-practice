import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import poisson
import pandas as pd
from scipy.stats import multinomial


#Gaussian distribution - f(x ) = (1/sqrt(2*pi()*(std() ** 2))) * exp(-((x-mean())**2)/(2*(std()**2)))
myu , sigma = 0, 1
x = np.linspace(-4, 4, 100)
y = (1/(np.sqrt(2*np.pi*(sigma**2))))*(np.exp(-1*((x - myu)**2)/(2*(sigma**2))))
plt.plot(x,y)
plt.title("Gaussian distribution")
plt.show()

#Bernoulli distribution
p = 0.6
plt.bar([0,1],[1-p, p], color="blue")
plt.title("Bernoulli distribution")
plt.xticks([0,1], labels=["0 (Failure)", "1 (Success)"])
plt.show()

#binomial distribution
n, p = 10, 0.5
x = np.arange(0, n+1)
print(x)
y = binom.pmf(x, n, p)
print(y)
plt.bar(x, y, color="green")
plt.title("Binomial distribution")
plt.show()

#Poisson distribution
lam = 3
x = np.arange(0, 10)
y = poisson.pmf(x, lam)
plt.bar(x, y, color="orange")
plt.title("Poisson Distribution")
plt.show()

# Problem statement: 
# - A disease affects 1% of population
# - A test is 95% accurate for diseased individuals and 90% accurate for non-diseased individuals
# - Find the probability of having the diseased, given the positive test outcome


#Bayes theorem
def bayes_theorem(prior, likelihood, evidence):
    return ((likelihood * prior) / evidence)

def bayes_theorem_compute(prior, sensitivity, specificity):
    evidence = (sensitivity * prior) + (1-specificity) * (1-prior)
    posterior = bayes_theorem(prior, sensitivity, evidence)
    return posterior

prior = 0.01 # Prior knowledge : 1% population gets affected by disease
sensitivity = 0.95 # True positive - 95%
specificity = 0.90 # True negative - 90%

posterior = bayes_theorem_compute(prior, sensitivity, specificity)
print("Probability of disease given positive tests =>", posterior)

#Problem
# Create a multinomial distribution graph based on multiclass data
# Load the drug sales data
df = pd.read_csv('sales_data.csv')

# Aggregate Units_Sold and Sales_Amount by Category
# This defines the outcomes of our multinomial trials
category_summary = df.groupby('Category').agg({
    'Units_Sold': 'sum',
    'Sales_Amount': 'sum'
}).reset_index()

# Calculate the probability parameters (p) for the Multinomial Distribution
units_total = category_summary['Units_Sold'].sum()
sales_total = category_summary['Sales_Amount'].sum()

category_summary['Units_Prob'] = category_summary['Units_Sold'] / units_total
category_summary['Sales_Prob'] = category_summary['Sales_Amount'] / sales_total

# Visualization using Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Multinomial Probabilities for Units Sold
ax1.bar(category_summary['Category'], category_summary['Units_Prob'], 
        color='skyblue', edgecolor='black', alpha=0.8)
ax1.set_title('Multinomial Distribution: Units Sold by Category', fontsize=14)
ax1.set_ylabel('Probability ($p_i$)', fontsize=12)
ax1.set_xlabel('Category', fontsize=12)
ax1.set_ylim(0, max(category_summary['Units_Prob']) * 1.2)
for i, v in enumerate(category_summary['Units_Prob']):
    ax1.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

# Plot 2: Multinomial Probabilities for Sales Amount
ax2.bar(category_summary['Category'], category_summary['Sales_Prob'], 
        color='salmon', edgecolor='black', alpha=0.8)
ax2.set_title('Multinomial Distribution: Sales Amount by Category', fontsize=14)
ax2.set_ylabel('Probability ($p_i$)', fontsize=12)
ax2.set_xlabel('Category', fontsize=12)
ax2.set_ylim(0, max(category_summary['Sales_Prob']) * 1.2)
for i, v in enumerate(category_summary['Sales_Prob']):
    ax2.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


