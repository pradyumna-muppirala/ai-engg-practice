import pandas as pd
import numpy as np

df = pd.read_csv("employee-data.csv")
print(df)

df.dropna()
print(df)


df2 =  pd.read_csv("employee-data.csv")
print(df2)

df.dropna(axis=1)
print(df2)

df2["email"] = df2["email"].fillna("a@b.com")
print(df2)
# df["salary"] = df["salary"].fillna(0)
df["salary"] = df["salary"].interpolate()
print(df)
# df.fillna(method="bfill")
# print(df)
print("renamed a column...")
df.rename(columns={"email":"email_address"})
print(df)
print("computed salary deviation")
df["salary_deviation"] = df["salary"] - df["salary"].mean()
print("treating salary column as float")
df["salary"] = df["salary"].astype(float)
print(df)

combined_df = pd.concat([df, df2], axis=0)
print(combined_df)

combined_df = pd.concat([df, df2], axis=1)
print(combined_df)

merged_df = pd.merge(df, df2 , on="email")
print(merged_df)

merged_df = pd.merge(df, df2 , how="left", on="email")
print(merged_df)
merged_df = pd.merge(df, df2 , how="right", on="email")
print(merged_df)
merged_df = pd.merge(df, df2 , how="inner" , on="email")
print(merged_df)

""" df.set_index("id")
df2.set_index("id")
joined= df.join(df2, how="inner")
pint(joined) """
# auto-generated code below
df.set_index("email", inplace=True)
df2.set_index("email", inplace=True)
joined = df.join(df2, how="inner", rsuffix="_df2")
print(joined)

print("Hands-on exercises....")
data = {"Name": ["Alice", "Bob", np.nan, "John"], "Age": ["20", np.nan, "30", "45"]}

df = pd.DataFrame(data)
print(df)
df["Age"] = df["Age"].astype(float)
df["Age"] = df["Age"].fillna(df["Age"].mean())
# df["Name"] = df["Name"].fillna(df["Name"].interpolate())
print(df)

df = df.rename(columns={"Name":"name", "Age":"age"})
print(df)

#one hot encoding example
df2_encoded = pd.get_dummies(df2["department"], prefix="department")
df2 = pd.concat([df2, df2_encoded], axis=1)
print(df2)