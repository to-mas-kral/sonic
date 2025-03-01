import numpy as np
import matplotlib.pyplot as plt

def function(lambd):
    return 0.003939804229 / np.power(np.cosh(0.0072 * (lambd - 538.0)), 2)

x = np.linspace(340, 830, 490)
x_func = np.vectorize(function)
y = x_func(x)

poly = np.polynomial.polynomial.Polynomial.fit(x, y, deg=7, domain=(360, 830))

#poly = poly.cutdeg(5)

print(poly)

plt.plot(x, y, x, poly(x))
plt.show()
