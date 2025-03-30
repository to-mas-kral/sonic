import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter

def set_common_plot_settings(ax, ylabel, typ, do_x_label=True):
    if (do_x_label):
        ax.set_xlabel(xlabel='Počet vzorků', labelpad=10)

    ax.set_ylabel(ylabel=ylabel, labelpad=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 32, 64, 128, 256, 512, 1024, 2048, 4096]))

    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.set_aspect('auto')

    if (typ == 'MSE'):
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

def create_lerp_chart(csv_ref_path, csv_test_path, outpath):
    ref_data = pd.read_csv(csv_ref_path, sep=",")
    test_data = pd.read_csv(csv_test_path, sep=",")

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(6, 12)
    #fig.tight_layout(pad=1.5)

    axs[0].plot(ref_data['Samples'], ref_data['MSE'], linewidth=1.3, label="Baseline")
    axs[0].plot(test_data['Samples'], test_data['MSE'], linewidth=1.3, label="Navádění")

    axs[1].plot(ref_data['Samples'], ref_data['FLIP'], linewidth=1.3, label="Baseline")
    axs[1].plot(test_data['Samples'], test_data['FLIP'], linewidth=1.3, label="Navádění")

    set_common_plot_settings(axs[0], 'MSE', 'MSE', False)
    set_common_plot_settings(axs[1], 'FLIP', 'FLIP')

    fig.savefig(outpath, bbox_inches='tight')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
    "font.size": 18,
})

# This is to keep ticks at min/max
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

#create_lerp_chart('./cornell-box-normal.csv', './cornell-box-lg.csv', 'MSE', "cornell-box-mse.pdf", "MSE")

create_lerp_chart('./cornell-box-f10-normal.csv', './cornell-box-f10-lg.csv', "cornell-box-f10-conv.pdf")
#create_lerp_chart('./cornell-box-f10-normal.csv', './cornell-box-f10-lg.csv', 'FLIP', "cornell-box-f10-flip.pdf", "FLIP")

create_lerp_chart('./staircase-normal.csv', './staircase-lg.csv', "staircase-conv.pdf")
#create_lerp_chart('./staircase-normal.csv', './staircase-lg.csv', 'FLIP', "staircase-flip.pdf", "FLIP")

create_lerp_chart('./machines-normal.csv', './machines-lg.csv', "machines-conv.pdf")
#create_lerp_chart('./machines-normal.csv', './machines-lg.csv', 'FLIP', "machines-flip.pdf", "FLIP")

plt.show()
