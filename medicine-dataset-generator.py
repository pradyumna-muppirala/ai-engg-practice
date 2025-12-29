import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define data parameters
n_rows = 1000
drugs = [
    {"id": "DRG001", "name": "Ibuprofen", "category": "Orthopaedics"},
    {"id": "DRG002", "name": "Amoxicillin", "category": "General"},
    {"id": "DRG003", "name": "Paracetamol", "category": "Paediatrics"},
    {"id": "DRG004", "name": "Tamsulosin", "category": "Urology"},
    {"id": "DRG005", "name": "Cephalexin", "category": "General"},
    {"id": "DRG006", "name": "Diclofenac", "category": "Orthopaedics"},
    {"id": "DRG007", "name": "Azithromycin", "category": "Paediatrics"},
    {"id": "DRG008", "name": "Finasteride", "category": "Urology"},
    {"id": "DRG009", "name": "Metformin", "category": "Endocrinology"},
    {"id": "DRG010", "name": "Ciprofloxacin", "category": "General"},
]

countries = ["USA", "UK", "Canada", "Australia", "Germany", "France", "India", "Japan"]

# Generate dataset
data = []
start_date = datetime(2023, 1, 1)

for i in range(n_rows):
    drug = drugs[np.random.randint(0, len(drugs))]
    data.append({
        "Drug_ID": drug["id"],
        "Drug_Name": drug["name"],
        "Category": drug["category"],
        "Country": np.random.choice(countries),
        "Units_Sold": np.random.randint(10, 500),
        "Sales_Amount": round(np.random.uniform(100, 5000), 2),
        "Date": start_date + timedelta(days=np.random.randint(0, 365))
    })

# Create DataFrame
df = pd.DataFrame(data)

# Display first few rows
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"\nDataset summary:\n{df.describe()}")

# Optionally save to CSV
df.to_csv("sales_data.csv", index=False)
print("\nData saved to sales_data.csv")