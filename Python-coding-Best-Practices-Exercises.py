#Pythonic coding examples
#1. Use descriptive variable names
#2. Allways write module code with functions and classes
#3. Handle file operations with proper exception handling
#4. Follow PEP8 style guidelines
#5. Avoid redundant code by using functions 
#6. Use context managers for file operations
#7. Implement error handling for file operations
#8. Use powerful Python built-in functions and libraries
#9. Write modular and reusable code
#10. Use lists and comprehensions for efficient data handling
# example: use list comprehension to create a list of squares for the first 10 natural numbers
squares = [x**2 for x in range(1, 11)]
print("Squares of the first 10 natural numbers:", squares)
#11. Document your code with comments and docstrings
#12. Test your code with different scenarios
#13. use lambda functions for small anonymous functions
#example: use a lambda function to sort a list of tuples based on the second element
data = [(1, 'apple'), (2, 'banana'), (3, 'cherry'), (4, 'date')]    
sorted_data = sorted(data, key=lambda x: x[1])
print("Sorted data:", sorted_data)
#14. Use f-strings for formatted output
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")
# use map and lambda to compute squares of a list of numbers
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x**2, numbers))
print("Squared numbers using map and lambda:", squared_numbers)
# example : function to filter based on a condition using filter and lambda
def filter_even_numbers(numbers):
    return list(filter(lambda x: x % 2 == 0, numbers))
even_numbers = filter_even_numbers(numbers)
print("Even numbers:", even_numbers)
# example : function to reduce a list of numbers to their product using reduce and lambda
from functools import reduce
def product_of_numbers(numbers):
    return reduce(lambda x, y: x * y, numbers)  
product = product_of_numbers(numbers)
print("Product of numbers:", product)
#example : use os module to show the current working directory and list the files in current directory
import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)
files_in_directory = os.listdir(current_directory)
print("Files in the current directory:", files_in_directory)
# example: use sys module to show the arguments and python version
import sys
print("Python version:", sys.version)
print("Command line arguments:", sys.argv)