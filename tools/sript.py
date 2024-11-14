 import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

x = [1, 2, 4, 6, 8, 10, 12]
y = [156.7, 80.5, 42.1, 28.5, 25.7, 23.4, 21.4]


optimal = [y[0] / scaling for scaling in x]

fig, ax = plt.subplots()

yline = plt.plot(x, y, 'o-')
optimalline = plt.plot(x, optimal, "o--")

plt.xticks(x);

plt.yscale('log', base=2)

plt.yticks(optimal)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.title("Multithreaded render scaling (12-core system)")

plt.ylabel("Render time")
plt.xlabel("Threads")

plt.legend( ["achieved", "optimal"])

plt.show()
