import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Latin Modern Roman",
    "font.size": 16,
})

macbethpb = "#505ba6"
data = pd.read_csv('spectral_data/macbeth_pb.csv')

fig = plt.figure(figsize=(6, 3.5))
ax = fig.add_subplot()

plot = ax.plot(data['wl'], data['sce'], color=macbethpb, linewidth=2)

ax.set_xlabel(xlabel='Vlnová délka (nm)', labelpad=10)
ax.set_ylabel(ylabel="\% odrazivost", labelpad=10)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.ylim(0, 100)
plt.xlim(400, 750)

ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([25, 50, 75, 100]))
ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([25, 50, 75, 100]))

fig.savefig("refl_macbethpb.pdf", bbox_inches='tight')

plt.show()

