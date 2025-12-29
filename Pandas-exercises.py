import pandas as pd

print("--- exercise 1 ---")
df3 = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
print(df3.head())
print(df3.tail(5))
print(df3.info())
print(df3.describe())

print(df3[df3["species"] == "setosa"])
df4 = df3[["species", "sepal_length"]]
print(df4)

# Additional exercises
print("--- exercise 1 ---")
df_employee = pd.read_csv("employee2.csv")
print(df_employee.head())
print(df_employee.describe())

print("--- exercise 2 ---")
friends_groups = { "firstfriendsgroup" : {"FirstCouple" : {"Name": ["Alice", "Bob"], "Age": ["20", "30"], "Gender": ["F", "M"]} , 
        "SecondCouple" : {"Name": ["John", "Jamey"], "Age": ["40", "35"], "Gender": ["M", "F"]} } , 
       "secondfriendsgroup" : {"FirstCouple" : {"Name": ["Jaya", "Raj"], "Age": ["25", "32"], "Gender": ["F", "M"]} , 
        "SecondCouple" : {"Name": ["Vijay", "Radha"], "Age": ["40", "35"], "Gender": ["M", "F"]} } }

df_friends = pd.json_normalize(friends_groups)
print(df_friends)

# Add third friends group with fictitious data
friends_groups["thirdfriendsgroup"] = {
    "FirstCouple": {"Name": ["Emma", "Chris"], "Age": ["28", "31"], "Gender": ["F", "M"]},
    "SecondCouple": {"Name": ["Sarah", "Michael"], "Age": ["38", "42"], "Gender": ["F", "M"]}
}

df_friends = pd.json_normalize(friends_groups)
print(df_friends)

df4 = pd.read_excel("employee-data.xlsx")
df5 = df4[df4["salary"] > 70000]
print(df5)
df5.to_csv("HNI_employees.csv")