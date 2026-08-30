
from math import sqrt
from random import randint

#Exercise 1
print("----- Exercise 1 -----" )  
#Integer and floats

age=20
height = 5.5

#Strings
name ="Alice"
greetings = "Hello " + name

#Lists and Tuples
numbers = [2,5,7,9]
names = ["Alice", "Bob", "Charlie"]
Coodinates = (34, 45)

#Dictionaries
person = { name: "Shiva", age:0 , height: 5.6}

#Booleans
is_human = False

print (age)
print (height)
print (name)
print (greetings)
print (numbers)
print (names)
print (Coodinates)
print (person)
print (is_human)

#Exercise 2
print("----- Exercise 2 -----" )    
#Define variables of different data types

Int_variable = 20
float_variable = 3.14
string_var = "AI"
list_var = [1, 2, 3, 4, 5]
tuple_var = (10, 20, 30)
dict_var = {"name": "Alice", "role": "AI engineer" }
bool_var = True
print(Int_variable)
print(float_variable)
print(string_var + " Bootcamp")
list_var.append(6)
print(list_var)
print(tuple_var)
print(dict_var)
print(bool_var)

#Exercise 3

print("----- Exercise 3 -----")

#conditional statements practice
if ( True == bool_var):
    print("Boolean variable is True")
else:
    print("Boolean variable is False")

if ( Int_variable > 50):
    print("Integer variable is greater than 50")
elif ( Int_variable == 50):
    print("Integer variable is equal to 50")
else:
    print("Integer variable is less than 50")   
age = 21
if (age >= 18 ):
    print("Eligible to vote")
    if (age < 30):
        print("Young voter")
    else:
        print("Adult voter")
else:
    print("Not eligible to vote")

#syntax for for-loop
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print("Fruit:", fruit)

for i in range(16):
    print("Square root of ", i, "is", sqrt(i))

#syntax for while-loop
count = 0
while count < 10:
    print("Count is:", count)
    count += 2

count = 0
while True:
    if count == 10:
        print("Breaking the infinite loop")
        break
    print("Infinite Loop Count:", count)
    count += 1

count = 0
while count < 10:
    count += 1
    if count % 2 == 0:
        continue
    print("Odd Count:", count)

for i in range(10):
    if i % 2 == 0:
        print("Even number => ", i)
    else:
        continue

n = randint(1, 100)
print("Random number generated is:", n)
divisor = 2
while (divisor == 2 or n % divisor != 0):
    if divisor > sqrt(n):
        print(n, "is a prime number")
        break
    divisor += 1
    if (n % divisor == 0):
        print(n, "is not a prime number")
        break

#Menu driven calculator
def add(x, y):
    return x + y
def subtract(x, y):
    return x - y
def multiply(x, y):
    return x * y
def divide(x, y):
    if (y == 0):
        print("Error: Division by zero")
        return None
    return x / y

while True:
    print("Menu:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Exit")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '5':
        print("Exiting the calculator.")
        break
    
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    result_label = "Result: "
    if choice == '1':
        print(result_label, add(num1, num2))
    elif choice == '2':
        print(result_label, subtract(num1, num2))
    elif choice == '3':
        print(result_label, multiply(num1, num2))      
    elif choice == '4':
        print(result_label, divide(num1, num2))
    else:
        print("Invalid choice. Please try again.")

n = int(input("Enter a number to calculate Factorial: "))
factorial = 1
for i in range(1, n + 1):
    factorial *= i
print("Factorial of", n, "is", factorial)

list_numbers = [randint(1, 100) for _ in range(10)]
print("List of random numbers:", list_numbers)
max_number = list_numbers[0]
for i in list_numbers:
    if i > max_number:
        max_number = i
print("Maximum number in the list is:", max_number)


