import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform 
# Random variable : dice roll

outcomes = [1, 2, 3, 4, 5, 6]
probabilities = np.array([1/6]*6)
print(probabilities)
# Expectations
expectation = np.sum(outcomes * probabilities)
print("Expectations (Mean):", expectation)

# Variance and Standard Deviation
variance = np.sum((outcomes - expectation)**2 * probabilities)
std_dev = np.sqrt(variance)

print("Variance : ", variance)
print("Standard Deviation: ", std_dev)

# simulating 10000 dice rolls
rolls = np.random.randint(1,7, size=10000)

#Calculate probabilities
P_even = np.sum((rolls %2 ==0)) / len(rolls)

P_greater_than_4 = np.sum((rolls > 4))/len(rolls)

print("Rolling an even number : ", P_even)
print("Rolling a number greater than 4 : ", P_greater_than_4)

""" plt.bar(outcomes, probabilities, color="blue")
plt.title("PMF of dice rolls")
plt.xlabel("Outcomes")
plt.ylabel("Probability")
plt.show() """

#Continuous random variable with uniform distribution.

""" x = np.linspace(0, 1, 100)
pdf = uniform.pdf(x, loc=0, scale=1)
plt.plot(x, pdf, color="red")
plt.title("PDF of uniform distribution")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show() """

#Simulate filling a coin 10000 times and compute probabilities of heads and tiles.

flips = np.random.randint(0,2, size=10000)
p_heads = sum((flips ==  1))/len(flips)
p_tails = sum((flips == 0))/len(flips)

print("Heads probability : ", p_heads, " Tails probability : ", p_tails)

#Probabilities of the weighted dice
weighted_probs = [float(1/7), float(1/6) , float(1/5), float(1/4), float(1/6), float(1/6)]
# Expectations
probabilities = np.array(weighted_probs)
print(probabilities)
expectation = np.sum(outcomes * probabilities)
print("Expectations (Mean):", expectation)

# Variance and Standard Deviation
variance = np.sum((outcomes - expectation)**2 * probabilities)
std_dev = np.sqrt(variance)

print("Variance : ", variance)
print("Standard Deviation: ", std_dev)
