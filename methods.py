# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_series(x):
#     result = x
#     a0 = x
#     i = 0

#     while abs(a0) >= 10**(-15) and i <= 100:
#         a0 *= -1 * x**2 *(2*i+1) /((2*i+3)**2 *(2*i+2))
#         result += a0
#         i += 1
#     return result

# def lagrange_L(x, u, i):
#     n = len(u)
#     mult = 1
#     for j in range(n):
#         if j == i:
#             continue
#         mult *= (x - u[j]) / (u[i] - u[j])
#     return mult

# def integral_sine(x, a, b):
#     f = lambda t: calculate_series(t)
#     result = 0
#     n = 10  # измените на 20, 40, 80...
#     u = inter_dots(a, b, n)
#     for i in range(len(u)):
#         L_i = f(u[i]) * lagrange_L(x, u, i)
#         result += L_i
#     return result

# def inter_dots(a, b, n):
#     x = []
#     for i in range(n+1):
#         if(i==0):
#             continue
#         x_i = a + (b - a) / (n - 1) * i
#         x.append(x_i)
#     return x

# x = np.linspace(0, 5, 11)  # точки x
# y = [0]
# s = []

# for x_i in x:
#   if x_i==0:
#     continue
#   integral_sin = integral_sine(x_i, 0, 5)  # вычисление интегрального синуса
#   y.append(integral_sin)
#   print(integral_sin)

# for x_i in x:
#   integral_sin = calculate_series(x_i)  # вычисление интегрального синуса
#   s.append(integral_sin)
#   print(integral_sin)

# plt.plot(x, y)
# plt.plot(x, np.abs(np.array(s) - np.array(y)))
# plt.xlabel('x')
# plt.ylabel('Integral Sine')
# plt.title('Integral Sine Function')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def calculate_series(x):
    result = x
    a0 = x
    i = 0

    while abs(a0) >= 10**(-15) and i <= 100:
        a0 *= -1 * x**2 *(2*i+1) /((2*i+3)**2 *(2*i+2))
        result += a0
        i += 1
    return result

def divided_differences(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    # Fill in the first column of the coef matrix with y values
    coef[:,0] = y
    
    # Compute the divided differences
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    
    return coef

def newton_interpolation(x, y, xi):
    n = len(x)
    coef = divided_differences(x, y)
    result = coef[0, 0]
    temp = 1.0
    
    for i in range(1, n):
        temp *= (xi - x[i - 1])
        result += coef[0, i] * temp
    
    return result

# Function for which to calculate the integral
def f(x):
    return calculate_series(x)

# Defining the interval of integration
a = 0
b = np.pi / 2

# Number of intervals
n = 10
max_iterations = 100
tolerance = 1e-7
integral_approximation_prev = 0

x = np.linspace(0, 5, 11)
y = [0]
s = []

for x_i in x:
  integral_sin = calculate_series(x_i)  # вычисление интегрального синуса
  s.append(integral_sin)
  print(integral_sin)

for x_i in x:
  if x_i==0:
    continue
  integral_sin = newton_interpolation(x[:n], s[:n], x_i)  # вычисление интегрального синуса
  y.append(integral_sin)
  print(integral_sin)

# Plot the graph of integral sin values calculated using Taylor series
plt.figure()
# plt.plot(x, s, label='Taylor series approximation')

# Plot the graph of integral sin values calculated using Newton interpolation
# plt.plot(x, y, label='Newton interpolation')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('Integral sin(x)')
# plt.title('Integral sin(x) approximations')
# plt.grid()
# plt.show()

# plt.plot(x, y)
plt.plot(x, np.abs(np.array(s) - np.array(y)))
plt.xlabel('x')
plt.ylabel('Integral Sine')
plt.title('Integral Sine Function')
plt.show()