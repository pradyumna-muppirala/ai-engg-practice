#Python practice for functions and other constructs and concepts
from random import randint
import math_operations as m 

print(m.sqrt(16))  # Example of using math module

def func1():
    print("Hello from func1!")

def func2(name):
    print(f"Hello, {name}!")

def add_numbers(num1, num2):
    return num1 + num2

result = add_numbers(5, 7)
print(f"The sum of 5 and 7 is: {result}")

#local scope example for variables in function

def greet():
    message = "Hello, World!"  # local variable
    print(message)

greet()

#global scope example for variables
Greeting = "Hi"

def greet_hello():
    print(Greeting)  # accessing global variable

greet_hello()
print(Greeting)  # accessing global variable outside function


n = randint(1, 10)
print(f"Random number between 1 and 10: {n}")
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print("The factorial of {n} is ",factorial(n))  

def is_even(num):
    return num % 2 == 0

if (is_even(n)):
    print(f"{n} is even: {is_even(n)}")
else:
    print(f"{n} is odd: {is_even(n)}")

import StringOps

s1 = "assa"
print(f"Reversed string of '{s1}': {StringOps.reverse_string(s1)}")
print(f"Uppercase of '{s1}': {StringOps.to_uppercase(s1)}")
print(f"Lowercase of '{s1}': {StringOps.to_lowercase(s1)}")
print(f"Number of vowels in '{s1}': {StringOps.count_vowels(s1)}")
print(f"Is '{s1}' a palindrome?: {StringOps.is_palindrome(s1)}")