import sympy as sp
import numpy as np

#derivatives
x = sp.Symbol('x')
derivative = sp.diff(x**2, x)
print(derivative)

#partial derivatives
y = sp.Symbol('y')
f = x**2 + y**2
partial_x = sp.diff(f, x)
partial_y = sp.diff(f, y)
print(partial_x)
print(partial_y)

# derivative of sin(x)
f_sin = sp.sin(x)
derivative_sin = sp.diff(f_sin, x)
print(derivative_sin)
# Evaluate derivative of sin(x) at x = 0
derivative_sin_at_0 = derivative_sin.subs(x, 0)
print(derivative_sin_at_0)

y = sp.Symbol('y')
f1 = 3*x**2 + 4*y**2 - 6*x*y
partial_x = sp.diff(f1, x)
partial_y = sp.diff(f1, y)
print(partial_x)
print(partial_y)

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradients = (1/m)* np.dot(X.T, errors)
        theta -= learning_rate * gradients
        if (0 == i%100):
            print("Gradients: ", gradients)
            print("Theta;", theta)
    return theta

X = np.array([[1,2], [2,3], [3,4]])
y = np.array([3, 3.5, 4])
theta = np.array([0.2, 0.2])
learning_rate = 0.1
iterations = 1000

optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations)
print(optimized_theta)

# Definite and indefinite integrals

x = sp.Symbol("X")
y= x**3
definite_integral = sp.integrate(y, (x, 0, 10))
indefinite_integral = sp.integrate(y, x)
print("Definite Integral => ", definite_integral)
print("Indefinite Integral => ", indefinite_integral)


x = sp.symbols('x')
f = sp.exp(-x)
area = sp.integrate(f, (x, 0, sp.oo))
print(area)  # Output: 1
indefinite_integral = sp.integrate(f, x)
print(indefinite_integral)

f1 = x**-1
area = sp.integrate(f1, (x, 0, sp.oo))
print(area)  # Output: 1
indefinite_integral = sp.integrate(f1, x)
print(indefinite_integral)

f1 = (2**x)**-1
area = sp.integrate(f1, (x, 0, sp.oo))
print(area)  # Output: 1
indefinite_integral = sp.integrate(f1, x)
print(indefinite_integral)

""" f2 = (1/x**(1/2+np.sqrt(-1)))
area = sp.integrate(f2, (x, 0, sp.oo))
print(area)  # Output: 1
indefinite_integral = sp.integrate(f2, x)
print(indefinite_integral) """

