import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

titanic_df = pd.read_csv("titanic.csv")
print(titanic_df.info())
print(titanic_df.describe())

#handling missing values of age
age_mean = titanic_df['age'].mean()
titanic_df.fillna({"age":age_mean}, inplace=True)
#remove duplicates , if any
titanic_df.drop_duplicates()
#filter data based on the classes
first_class_passengers = titanic_df[titanic_df['pclass'] == 1]
print(first_class_passengers)

#Bar chart : Survival by class

Survival_by_class = titanic_df.groupby("pclass")["survived"].mean()
Survival_by_class.plot(kind="bar", color="green")
plt.ylabel("Survival rate")
plt.title("Survival rate by class")
plt.show()

#Histogram of age distribution
titanic_df['age'].plot(kind="hist", bins=30, color="blue", edgecolor="black")
plt.xlabel("age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()

#Scarter plot of Age vs Fare 
sbn.scatterplot(data=titanic_df, x="age", y="fare", hue="survived", palette="Set1")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.show()