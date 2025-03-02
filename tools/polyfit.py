import numpy as np
import matplotlib.pyplot as plt

def pdf(lambd):
    return 0.003939804229 / np.power(np.cosh(0.0072 * (lambd - 538.0)), 2)

def inversecdf(xi):
    return 538.0 - 138.888889 * np.atanh(0.85691062 - 1.82750197 * xi);

pdf_func = np.vectorize(pdf)
inversecdf_func = np.vectorize(inversecdf)

x_lambd = np.linspace(340, 830, 490)
y_lambd = pdf_func(x_lambd)

poly = np.polynomial.polynomial.Polynomial.fit(x_lambd, y_lambd, deg=7, domain=(360, 830))

print("Polynomial")

print(poly)

print("Integral")

integral = poly.integ()

print(integral)

#plt.plot(x_lambd, y_lambd, x_lambd, poly(x_lambd))

x_xi = np.linspace(0, 1, 1000)
y_inversecdf = inversecdf(x_xi)

plt.plot(x_lambd, y_lambd, x_lambd, integral(x_lambd))

#plt.plot(x_xi, y_inversecdf, x_xi, integral(x_xi))

plt.show()

"""0.78657379(x) - 0.49603113(x)^2 - 0.42978345(x)^3 + 0.60793687(x)^4 + 0.17215255(x)^5 - 0.40395113(x)^6 - 0.02892187(x)^7 + 0.11084562(x)^8"""