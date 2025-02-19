import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
    "font.size": 12,
})

def create_chart(csv_path, ykey, outpath, color, ylabel, transform, kind):
    data = pd.read_csv(csv_path)

    # Corresponds to 8cm width
    fig = plt.figure(figsize=(3.1496062992, 1.8372703412))
    ax = fig.add_subplot()

    plot = ax.plot(data['wl'], data[ykey].transform(transform), color=color, linewidth=2)

    ax.set_xlabel(xlabel='Vlnová délka (nm)', labelpad=10)
    ax.set_ylabel(ylabel=ylabel, labelpad=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlim(400, 750)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([400, 450, 500, 550, 600, 650, 700, 750]))

    if kind == "refl":
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0., 0.25, 0.50, 0.75, 1.0]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0., 0.25, 0.50, 0.75, 1.0]))
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    fig.savefig(outpath, bbox_inches='tight')
    #plt.show()

def create_multi_chart(csv_path, ykeys, legend_labels, outpath, colors, ylabel, transform, kind):
    data = pd.read_csv(csv_path)

    # Corresponds to 8cm width
    fig = plt.figure(figsize=(3.1496062992, 1.8372703412))
    ax = fig.add_subplot()

    for key, color, legend_label in zip(ykeys, colors, legend_labels):
        ax.plot(data['wl'], data[key].transform(transform), color=color, linewidth=2, label=legend_label)

    ax.set_xlabel(xlabel='Vlnová délka (nm)', labelpad=10)
    ax.set_ylabel(ylabel=ylabel, labelpad=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlim(400, 750)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([400, 450, 500, 550, 600, 650, 700, 750]))

    if kind == "refl":
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0., 0.25, 0.50, 0.75, 1.0]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0., 0.25, 0.50, 0.75, 1.0]))
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.legend(frameon=False)

    fig.savefig(outpath, bbox_inches='tight')
    #plt.show()

create_chart('spectral_data/macbeth_pb.csv', 'val', "refl_macbethpb.pdf", "#505ba6", "Odrazivost", lambda x: x / 100., "refl")
create_chart('spectral_data/CIE_std_illum_D65.csv', 'val', "illum_d65.pdf", "#505ba6", "Relativní záře", lambda x : x, "illum")
create_chart('spectral_data/CIE_illum_FLs.csv', 'val10', "illum_fl10.pdf", "#505ba6", "Relativní záře", lambda x : x, "illum")

create_multi_chart('spectral_data/CIE_xyz_1931_2deg.csv', ['x', 'y', 'z'], ['X', 'Y', 'Z'], "cm_xyz.pdf", ["Red", "Green", "Blue"], "Relativní senzitivita", lambda x : x, "sens")
