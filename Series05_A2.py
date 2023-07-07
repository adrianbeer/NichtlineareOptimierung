import numpy as np
from functools import partial


def rosenbrock(x, a, b):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


def gradient(f, x, h):
    # x elem R^n
    # f: R^n -> R
    # d: direction
    # h: small
    n_dim = len(x)
    grad = np.ndarray((n_dim,))
    for p in range(n_dim):
        direction = np.zeros(n_dim)
        direction[p] = 1
        grad[p] = (f(x + direction*h) - f(x)) / h
    return grad

#def hesse_matrix(f, x, d, ,h):


def derivative(f, x, d, h):
    return (f(x + d * h) - f(x)) / h


def armijo_holds(f, x_curr, x_next, direction, nu, alpha, h):
    return f(x_next) <= f(x_curr) + alpha * nu * derivative(f, x_curr, direction, h)


def newton_step(f, x_curr):
    # Armijo parameters
    alpha = 1
    nu = 0.3
    h = 0.001

    direction = - gradient(f, x_curr, h)
    step = direction * alpha * nu

    x_next = x_curr + step

    # Armijo
    while not armijo_holds(f, x_curr, x_next, direction, nu, alpha, h):
        alpha = alpha/2
        step = direction * alpha * nu
        x_next = x_curr + step
    return x_next


def newton_opt(f, x_start, n):
    x = x_start
    for i in range(n):
        grad = gradient(f, x, h=0.001)
        print(f"Function value {f(x)} at {x} w/ gradient {grad}")
        x = newton_step(f, x)

a=1
b=100
newton_opt(partial(rosenbrock, a=a, b=b), x_start=(0.9,0.8), n=500)
print(f"Golbal minimum at {(a, a**2)}")


