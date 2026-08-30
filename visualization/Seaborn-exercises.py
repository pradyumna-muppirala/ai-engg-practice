import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sales_data = pd.read_csv('sales_data.csv')

""" data = np.random.rand(5,5)
sns.heatmap(data, annot=True,  cmap="coolwarm")
plt.title("Heat map")
plt.show()


sns.pairplot(sales_data)
plt.title("Pair plot")
plt.show() """

""" sales_by_date = sales_data.groupby('Date')['Sales_Amount'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_by_date, x='Date', y='Sales_Amount')
plt.title("Sales Over Time")
plt.xlabel("Date Time")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sales_by_drug = sales_data.groupby('Drug_Name')['Sales_Amount'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=sales_by_drug, x='Drug_Name', y='Sales_Amount')
plt.title("Total Sales Amount by Drug Name")
plt.xlabel("Drug Name")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sales_by_drug_units = sales_data.groupby('Drug_Name')['Units_Sold'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.scatterplot(data=sales_by_drug_units, x='Drug_Name', y='Units_Sold')
plt.title("Total Units Sold by Drug Name")
plt.xlabel("Drug Name")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 """
""" sales_data2 = sales_data
del sales_data2['Drug_ID']
del sales_data2['Drug_Name']
del sales_data2["Category"]
del sales_data2["Country"]
del sales_data2["Date"]

correlation_matrix = sales_data2.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()
 """
plt.figure(figsize=(12, 6))
units_by_drug_country = sales_data.groupby(['Drug_Name', 'Country'])['Units_Sold'].sum().unstack(fill_value=0)
units_by_drug_country.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Total Units Sold by Drug Name - All Countries")
plt.xlabel("Drug Name")
plt.ylabel("Units Sold")
plt.legend(title="Country")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=sales_data, x='Drug_Name', y='Units_Sold')
plt.title("Distribution of Units Sold by Drug Name")
plt.xlabel("Drug Name")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Sales by Category
sales_by_category = sales_data.groupby('Category')['Sales_Amount'].sum()
axes[0, 0].bar(sales_by_category.index, sales_by_category.values, color='skyblue')
axes[0, 0].set_title('Total Sales by Category')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Sales Amount')
axes[0, 0].tick_params(axis='x', rotation=45)

# Subplot 2: Units Sold by Drug Name
units_by_drug = sales_data.groupby('Drug_Name')['Units_Sold'].sum().sort_values(ascending=False).head(10)
axes[0, 1].barh(units_by_drug.index, units_by_drug.values, color='lightcoral')
axes[0, 1].set_title('Top 10 Drugs by Units Sold')
axes[0, 1].set_xlabel('Units Sold')

# Subplot 3: Sales Over Countries
sales_by_country = sales_data.groupby('Country')['Sales_Amount'].sum().sort_values(ascending=False)
axes[1, 0].pie(sales_by_country.values, labels=sales_by_country.index, autopct='%1.1f%%')
axes[1, 0].set_title('Sales Distribution by Country')

# Subplot 4: Average Units Sold by Category
avg_units_by_category = sales_data.groupby('Category')['Units_Sold'].mean()
axes[1, 1].plot(avg_units_by_category.index, avg_units_by_category.values, marker='o', linewidth=2, markersize=8, color='green')
axes[1, 1].set_title('Average Units Sold by Category')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Average Units Sold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()