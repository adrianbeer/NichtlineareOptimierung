import scipy
import numpy as np


def f(x):
    a, b = x[0], x[1]
    return (a**2 + b - 11)**2 + (a  + b**2 - 7)**2

def g(x):
    if x[0] >= np.abs(x[1]):
        return 5*np.sqrt(9*x[0]**2 + 16*x[1]**2)
    elif x[0] > 0 and x[0] < np.abs(x[1]):
        return 9*x[0] + 16*np.abs(x[1])
    else:
        return 9*x[0] + 16*np.abs(x[1]) - x[0]**9       



res = scipy.optimize.minimize(g, (0.1, 0.1), method='SLSQP', bounds=((-0.5, 10000), (-10000, 10000)))
# print(res)

res = scipy.optimize.differential_evolution(g,  bounds=((-0.5, 10000), (-10000, 10000)))
print(res)