import matplotlib.pyplot as plt
import pandas  as pd

#Basic plot
x = [1 ,2 ,3 ,4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.show()

# Line plot with labels and legend
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y, label='Data Series')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot Title')
plt.legend()
plt.show() 

# Bar chart for student marks by grade
grades = ['A', 'B', 'C']
math_marks = [90, 75, 60]

plt.bar(grades, math_marks)
plt.xlabel('Grades')
plt.ylabel('Mathematics Marks')
plt.title('Student Marks by Grade in Mathematics')
plt.show()


# Read the CSV file
sales_data = pd.read_csv('sales_data.csv')

# Histogram of medicine purchases
plt.figure(figsize=(10, 5))
plt.hist(sales_data['Sales_Amount'], bins=20, edgecolor='black')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Medicine Purchases')
plt.show()

# Stacked histogram by country
countries = sales_data['Country'].unique()
plt.figure(figsize=(12, 6))

for country in countries:
    country_data = sales_data[sales_data['Country'] == country]['Sales_Amount']
    plt.hist(country_data, bins=20, alpha=0.6, label=country, edgecolor='black')

plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of Sales by Country')
plt.legend()
plt.show()


#Scatter plot
x= [1, 2, 3, 4, 5]
y= [10, 12, 25, 30, 40]

plt.scatter(x, y)
plt.title("Scatter plot", color="green")
plt.show()