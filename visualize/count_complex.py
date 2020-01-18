import csv
import os
from collections import defaultdict

import numpy as np

from config import RESULTS_DIR
from dataset import PATIENTS
from utils import make_dirs
from visualize.common import Result


def is_complex(array, counter):
    if array.dtype == np.complex:
        if np.all(np.imag(array) == 0):
            # complex number however with zero imaginary part
            counter['complex_zero'] += 1
        else:
            counter['complex'] += 1
    else:
        counter['real'] += 1


counter = {}

artifact_removal_name = 'peak'
for n_csp in list(range(2, 22, 2)):  # full unmixing matrix is same for all #CSP components
    results = {}
    for patient_name in PATIENTS:
        results[patient_name] = Result(patient_name, artifact_removal_name)
        counter[patient_name] = defaultdict(dict)

        for csp_method_name, result in results[patient_name].results.items():
            counter[patient_name][csp_method_name]['w'] = defaultdict(int)
            counter[patient_name][csp_method_name]['eigenvalues'] = defaultdict(int)
            counter[patient_name][csp_method_name]['full_w'] = defaultdict(int)

            for r in result:
                is_complex(r[n_csp]['W_T'], counter[patient_name][csp_method_name]['w'])
                if 'eigenvalues' in r[n_csp] and r[n_csp]['eigenvalues'] is not None:
                    is_complex(r[n_csp]['eigenvalues'], counter[patient_name][csp_method_name]['eigenvalues'])
                if 'full_W_T' in r[n_csp] and r[n_csp]['full_W_T'] is not None:
                    is_complex(r[n_csp]['full_W_T'], counter[patient_name][csp_method_name]['full_w'])

    output_path = os.path.join(RESULTS_DIR, 'complex')
    make_dirs(output_path)

    def write_csv(writer, key):
        # header
        row = ['Method']
        for patient_name in PATIENTS:
            row.append('{}: real'.format(patient_name))
            row.append('{}: complex'.format(patient_name))
            row.append('{}: 0-complex'.format(patient_name))
        row.append('Group: real')
        row.append('Group: complex')
        row.append('Group: 0-complex')
        writer.writerow(row)

        for csp_method_name in counter[PATIENTS[0]].keys():
            row = [csp_method_name]

            for patient_name in PATIENTS:
                row.append(counter[patient_name][csp_method_name][key]['real'])
                row.append(counter[patient_name][csp_method_name][key]['complex'])
                row.append(counter[patient_name][csp_method_name][key]['complex_zero'])

            row.append(np.sum([counter[patient_name][csp_method_name][key]['real'] for patient_name in PATIENTS]))
            row.append(np.sum([counter[patient_name][csp_method_name][key]['complex'] for patient_name in PATIENTS]))
            row.append(np.sum([counter[patient_name][csp_method_name][key]['complex_zero'] for patient_name in PATIENTS]))

            writer.writerow(row)


    with open(os.path.join(output_path, 'w_counter_{}.csv'.format(n_csp)), 'w') as f:
        write_csv(csv.writer(f), 'w')
    with open(os.path.join(output_path, 'eigenvalues_counter_{}.csv'.format(n_csp)), 'w') as f:
        write_csv(csv.writer(f), 'eigenvalues')
    with open(os.path.join(output_path, 'full_w_{}.csv'.format(n_csp)), 'w') as f:
        write_csv(csv.writer(f), 'full_w')
