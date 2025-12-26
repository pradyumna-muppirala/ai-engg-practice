
def add(num1, num2):
    return num1 + num2

def subtract(num1, num2):
    return num1 - num2

def multiply(num1, num2):
    return num1 * num2

def divide(num1, num2):
    if num2 == 0:
        return "Error! Division by zero."
    return num1 / num2

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def power(base, exponent):
    return base ** exponent

def sqrt(n):
    if n < 0:
        return "Error! Cannot compute square root of negative number."
    return n ** 0.5

def modulus(num1, num2):
    return num1 % num2

def floor_divide(num1, num2):
    if num2 == 0:
        return "Error! Division by zero."
    return num1 // num2

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True 

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def absolute(n):
    return abs(n)

def logarithm(n, base=10):
    import math
    if n <= 0:
        return "Error! Logarithm undefined for non-positive numbers."
    return math.log(n, base)
    
