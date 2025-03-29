import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter

def set_common_plot_settings(ax, ylabel, typ):
    ax.set_xlabel(xlabel='Počet vzorků', labelpad=10)
    ax.set_ylabel(ylabel=ylabel, labelpad=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 32, 64, 128, 256, 512, 1024, 2048, 4096]))

    if (typ == "MSE"):
        ax.set_yscale('log', base=10)
        ax.set_xlim(left=1)
        #ax.set_ylim(bottom=1)
    else:
        ax.set_yscale('log', base=10)
        ax.set_xlim(left=1)
        #ax.set_ylim(bottom=1)

    ax.set_xscale('log', base=2)

    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.legend(frameon=False)

def create_lerp_chart(csv_ref_path, csv_test_path, ykey, outpath, ylabel):
    ref_data = pd.read_csv(csv_ref_path, sep=",")
    test_data = pd.read_csv(csv_test_path, sep=",")

    # Corresponds to 8cm width
    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot()

    ax.plot(ref_data['Samples'], ref_data[ykey], linewidth=1.3, label="Baseline")
    ax.plot(test_data['Samples'], test_data[ykey], linewidth=1.3, label="Navádění")

    set_common_plot_settings(ax, ylabel, ykey)

    fig.savefig(outpath, bbox_inches='tight')

plt.rcParams.update({
    "text.usetex": True,    
    "font.family": "Computer Modern Roman",
    "font.size": 22,
})

# This is to keep ticks at min/max
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

create_lerp_chart('./cornell-box-normal.csv', './cornell-box-lg.csv', 'MSE', "cornell-box-mse.pdf", "MSE")
create_lerp_chart('./cornell-box-f10-normal.csv', './cornell-box-f10-lg.csv', 'MSE', "cornell-box-f10-mse.pdf", "MSE")
create_lerp_chart('./cornell-box-f10-normal.csv', './cornell-box-f10-lg.csv', 'FLIP', "cornell-box-f10-flip.pdf", "FLIP")

plt.show()
