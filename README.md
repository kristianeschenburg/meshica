## What is meshICA?

A set of python scripts that implement group-ICA for graph-based data.  Also included is a dual-regression method to map the group-ICA components back onto single-subject time-series.

(Free software: 3-clause BSD license)

## Included Algorithms:
* [CanonicalICA](https://www.ncbi.nlm.nih.gov/pubmed/20153834) (as implemented in Scikit-Learn)
* [MELODIC'S Incremental Group-PCA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4289914/) (MIGP)

## How to install and use:

```bash
git clone https://github.com/kristianeschenburg/meshica.git
cd  ./meshica
pip install .
```

```python
from meshICA import canica, migp, dual_regression
from glob import glob

# for both the CanICA and MIGP classes, we need to provide their fit methods with a list of files
# these can be either '.mat' files or '.gii' files.

path = 'data_path/'
input_files = glob.glob(''.join([path,'*.mat']))

C = canica.CanICA()
C.fit(input_files=input_files)

M = migp.MIGP(input_files=input_files)
```

Both ```MIGP``` and ```CanICA``` generate an attribute called ```components_```, which are the group-level ICA components.  We can feed these back into the dual-regression algorithm as follows:

```python
D = dual_regression.DualRegression()
# for CanICA components
D.fit(input_files=input_files, C.components_)

# for MIGP components
D.fit(input_files=input_files, M.components_)
```