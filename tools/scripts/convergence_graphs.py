import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
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

def create_lerp_chart(csv_ref_path, csv_test_path, outpath, id_landscape=False):
    n_rows = 2
    n_cols = 1

    width = 6
    height = 10

    if (id_landscape):
        n_rows = 1
        n_cols = 2

        width = 12
        height = 6

    fig, axs = plt.subplots(n_rows, n_cols, layout="constrained")
    fig.set_size_inches(width, height)
    #fig.tight_layout(pad=1.5)

    ref_data, test_data = load_dataframes(csv_ref_path, csv_test_path)

    axs[0].plot(ref_data['Samples'], ref_data['MSE'], linewidth=1.3, label="Baseline")
    axs[0].plot(test_data['Samples'], test_data['MSE'], linewidth=1.3, label="Navádění")

    axs[1].plot(ref_data['Samples'], ref_data['FLIP'], linewidth=1.3, label="Baseline")
    axs[1].plot(test_data['Samples'], test_data['FLIP'], linewidth=1.3, label="Navádění")

    set_common_plot_settings(axs[0], 'MSE', 'MSE', id_landscape)
    set_common_plot_settings(axs[1], 'FLIP', 'FLIP')

    fig.savefig(outpath, bbox_inches='tight')

def load_dataframes(csv_ref_path, csv_test_path):
    ref_data = pd.read_csv(csv_ref_path, sep=",")
    test_data = pd.read_csv(csv_test_path, sep=",")

    min_len = min(len(ref_data['Samples']), len(test_data['Samples']))

    ref_data = pd.DataFrame({
        'Samples': ref_data['Samples'][:min_len].reset_index(drop=True),
        'MSE': ref_data['MSE'][:min_len].reset_index(drop=True),
        'FLIP': ref_data['FLIP'][:min_len].reset_index(drop=True)
    })

    test_data = pd.DataFrame({
        'Samples': test_data['Samples'][:min_len].reset_index(drop=True),
        'MSE': test_data['MSE'][:min_len].reset_index(drop=True),
        'FLIP': test_data['FLIP'][:min_len].reset_index(drop=True)
    })

    return ref_data,test_data

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
    "font.size": 18,
})

# This is to keep ticks at min/max
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

create_lerp_chart('./csvs/cboxf-normal.csv', './csvs/cboxf-lg.csv', "./pdfs/cornell-box-f10-conv.pdf")
create_lerp_chart('./csvs/cbox-normal.csv', './csvs/cbox-lg.csv', "./pdfs/cornell-box-conv.pdf")
create_lerp_chart('./csvs/kitchenpc-normal.csv', './csvs/kitchenpc-lg.csv', "./pdfs/kitchenpc-conv.pdf")
create_lerp_chart('./csvs/staircaseph-normal.csv', './csvs/staircaseph-lg.csv', "./pdfs/staircaseph-conv.pdf")
#create_lerp_chart('./csvs/machines-normal.csv', './csvs/machines-lg.csv', "./pdfs/machines-conv.pdf")

plt.show()
