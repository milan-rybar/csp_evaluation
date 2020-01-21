"""
Plot figures for the paper.
"""

from matplotlib import pyplot as plt
plt.style.use('paper_style')
import pandas as pd
import seaborn as sns
from dataset import PATIENTS
from visualize.common import Result
import brewer2mpl
from matplotlib.collections import LineCollection

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
# plt.rc('font', size=8)  # all
# plt.rc('xtick', labelsize=6)
# plt.rc('ytick', labelsize=6)
plt.rc('axes', titlesize=15)

artifact_removal_name = 'peak'
classifier_name = 'lda'


# load all results
results = {}
for patient_name in PATIENTS:
    results[patient_name] = Result(patient_name, artifact_removal_name)


def fill_data(method, name, data_for_pandas):
    for n_csp in results['aa'].n_csp_components:
        for patient_name in results.keys():
            for r in results[patient_name].get_results(method, n_csp, classifier_name):
                data_for_pandas.append((name, n_csp, r * 100.0))


#%%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.3 * 1.6, 2.7 * 1.6))

# GAP
ax = axes[0]
data_for_pandas = []
fill_data('gap_eig', 'Python (\\textit{eig})', data_for_pandas)
fill_data('matlab_gap_no_check', 'Matlab', data_for_pandas)
fill_data('gap_dr', 'PCA$\\rightarrow$(Python, Matlab)', data_for_pandas)
fill_data('fieldtrip', 'Fieldtrip, BBCI', data_for_pandas)

df = pd.DataFrame(data=data_for_pandas, columns=['method', 'n_csp', 'acc'])
sns.pointplot(x='n_csp', y='acc', hue='method', data=df, ax=ax,
              dodge=0.3, ci=95, capsize=0.2, scale=0.5, legend=False,
              # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
              linestyles=['--', '--', '-', ':'], errwidth=1,
              palette=brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors)
ymin1, ymax1 = ax.get_ylim()

# add significance line
def add_sig_line(x_start, x_end, y, ax, y_offset=1):
    lc = LineCollection([
        [(x_start, y), (x_end, y)],
        [(x_start, y - y_offset), (x_start, y)],
        [(x_end , y - y_offset), (x_end, y)]
    ], color=['k'], lw=1)
    ax.add_collection(lc)

add_sig_line(x_start=-0.3, x_end=9.3, y=90, ax=ax)
# ax.text(0, 91, '{\small $p < 0.0001$}')
ax.text(4.2, 90.2, '{\\small ****}')

ax.set_xlabel('Number of CSP components')
ax.set_ylabel('Accuracy')
ax.set_title('Geometric approach')
ax.legend(loc='lower right', scatterpoints=1, handlelength=1, bbox_to_anchor=(1.03, 0))#, borderaxespad=0.3)


# GEP
ax = axes[1]
data_for_pandas = []
fill_data('gep_no_checks', 'Python (\\textit{eig})', data_for_pandas)
fill_data('matlab_gep_no_check', 'Matlab', data_for_pandas)
fill_data('pca_gep', 'PCA$\\rightarrow$(Python (\\textit{eigh}), Matlab)', data_for_pandas)
fill_data('pca_gep_no_checks', 'PCA$\\rightarrow$ Python (\\textit{eig})', data_for_pandas)
fill_data('pca_mne', 'PCA$\\rightarrow$ MNE', data_for_pandas)

df = pd.DataFrame(data=data_for_pandas, columns=['method', 'n_csp', 'acc'])
sns.pointplot(x='n_csp', y='acc', hue='method', data=df, ax=ax,
              dodge=0.3, ci=95, capsize=0.2, scale=0.5, legend=False,
              linestyles=['--', '--', '-', '-',  ':'], errwidth=1,
              palette=brewer2mpl.get_map('Dark2', 'qualitative', 5).mpl_colors)
ymin2, ymax2 = ax.get_ylim()


# add significance line
add_sig_line(x_start=-0.3, x_end=4.5, y=90, ax=ax)
# ax.text(0, 91, '{\small $p < 0.0001$}')
ax.text(2, 90.2, '{\\small ****}')
add_sig_line(x_start=4.5, x_end=5.5, y=90, ax=ax)
# ax.text(4.5, 91, '{\small $p < 0.001$}')
ax.text(4.8, 90.2, '{\\small ***}')
add_sig_line(x_start=5.5, x_end=6.5, y=90, ax=ax)
# ax.text(6, 91, '{\small $p < 0.01$}')
ax.text(5.88, 90.2, '{\\small **}')


ax.set_xlabel('Number of CSP components')
ax.set_ylabel('Accuracy')
ax.set_title('Generalized eigenvalue problem')
ax.legend(loc='best', scatterpoints=1, handlelength=1)#, borderaxespad=0.3)


# share the same Y scale
axes[0].set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
axes[1].set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))

fig.savefig('peak_accuracy_lda.pdf')#, bbox_inches='tight')
plt.close(fig)
