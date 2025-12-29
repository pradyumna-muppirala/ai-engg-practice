import pandas as pd

s1 = pd.Series([1, 2, 3], index=["a", "b", "c"] )
print (s1)

data = { "firstfriendsgroup" : {"FirstCouple" : {"Name": ["Alice", "Bob"], "Age": ["20", "30"], "Gender": ["F", "M"]} , 
        "SecondCouple" : {"Name": ["John", "Jamey"], "Age": ["40", "35"], "Gender": ["M", "F"]} } , 
       "secondfriendsgroup" : {"FirstCouple" : {"Name": ["Jaya", "Raj"], "Age": ["25", "32"], "Gender": ["F", "M"]} , 
        "SecondCouple" : {"Name": ["Vijay", "Radha"], "Age": ["40", "35"], "Gender": ["M", "F"]} } }
df = pd.DataFrame(data)
print(df)
print("---- Reading CSV file -----")
df1 = pd.read_csv("employee-data.csv")
print(df1)
print("----- Reading Excel file -----")
df2 = pd.read_excel("employee-data.xlsx")
print(df2)

df2.to_csv("employee2.csv")

#Viewing data
print(df1.head())
print(df2.tail(5))
print(df2.info())
print(df2.describe())

print(df2[["name", "salary"]])
print(df2[df2["salary"] > 70000])
print(df2.iloc[0])
print(df2.loc[0])
print(df2.loc[:, ["name", "salary"]])