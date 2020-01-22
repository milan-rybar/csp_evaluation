# Evaluation of common spatial patterns implementations for covariance matrices without full rank

### Content

- `implementations`: CSP implementations
    - Our implementations:
        - `csp_python.py`: all Python implementations
        - `csp_geometric_approach_dim_reduction.m`: Matlab geometric approach with dimensionality reduction during the whitening step
        - `csp_geometric_approach_no_checks.m`: Matlab geometric approach without dimensionality reduction (no checks)
        - `csp_gep_no_checks.m`: Matlab generalized eigenvalue problem approach (no checks)
    - Toolboxes:
        - `use_bbci.m`: BBCI Toolbox
        - `use_fieldtrip.m`: FieldTrip
        - `use_biosig.m`: BioSig 
        - `use_mne.py`: MNE
- `artifacts_removal`: artifact removal methods based on ICA
- `evaluation.py`: Evaluate all CSP implementations on a binary motor imagery classification task


## Replication

Create the Conda environment from the `environment.yml` file:
```console
conda env create -f environment.yml
```
Activate the new environment:
```console
conda activate csp_evaluation
```

Note that CSP implementations from Matlab Toolboxes for EEG analysis
contain hard-coded paths for toolboxes locations in their corresponding .m files.

Directory `results` already contains pre-computed ICA for each patient.

Evaluation can be run in parallel (on the same or multiple computers), which is useful for Matlab implementations, 
by uncommenting line 166 in `evaluation.py`
```python
# @TaskGenerator
```
Then, tasks will be executed using [jug](https://jug.readthedocs.io/en/latest/) package.
Run it by 
```console
jug execute evaluation.py
```
and see the progress by
```console
jug status evaluation.py
```

Results can be inspected and visualized by scripts in `visualize` directory.
