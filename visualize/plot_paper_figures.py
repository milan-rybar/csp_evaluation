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

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
# plt.rc('font', size=8)  # all
# plt.rc('xtick', labelsize=6)
# plt.rc('ytick', labelsize=6)

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
fill_data('gap_eig', 'Python (eig)', data_for_pandas)
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

ax.set_xlabel('Number of CSP components')
ax.set_ylabel('Accuracy')
ax.set_title('Geometric approach')
ax.legend(loc='best', scatterpoints=1, handlelength=1)#, borderaxespad=0.3)


# GEP
ax = axes[1]
data_for_pandas = []
fill_data('gep_no_checks', 'Python (eig)', data_for_pandas)
fill_data('matlab_gep_no_check', 'Matlab', data_for_pandas)
fill_data('pca_gep', 'PCA$\\rightarrow$(Python (eigh), Matlab)', data_for_pandas)
fill_data('pca_gep_no_checks', 'PCA$\\rightarrow$ Python (eig)', data_for_pandas)
fill_data('pca_mne', 'PCA$\\rightarrow$ MNE', data_for_pandas)

df = pd.DataFrame(data=data_for_pandas, columns=['method', 'n_csp', 'acc'])
sns.pointplot(x='n_csp', y='acc', hue='method', data=df, ax=ax,
              dodge=0.3, ci=95, capsize=0.2, scale=0.5, legend=False,
              linestyles=['--', '--', '-', '-',  ':'], errwidth=1,
              palette=brewer2mpl.get_map('Dark2', 'qualitative', 5).mpl_colors)
ymin2, ymax2 = ax.get_ylim()

ax.set_xlabel('Number of CSP components')
ax.set_ylabel('Accuracy')
ax.set_title('Generalized eigenvalue problem')
ax.legend(loc='best', scatterpoints=1, handlelength=1)#, borderaxespad=0.3)


# share the same Y scale
axes[0].set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
axes[1].set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))

fig.savefig('/home/milan/research-notes/csp_paper/paper/data/peak_accuracy_lda.pdf')#, bbox_inches='tight')
plt.close(fig)
