## What is meshICA?

A set of python scripts that implement group-ICA for graph-based data, along with dual-regression to map the group-ICA components back onto single-subject time-series.

(Free software: 3-clause BSD license)

## Included Algorithms:
* [CanonicalICA](https://www.ncbi.nlm.nih.gov/pubmed/20153834) (as implemented in Scikit-Learn)
* [MELODIC'S Incremental Group-PCA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4289914/) (MIGP)

## How to install and use:

```bash
git clone https://github.com/kristianeschenburg/meshICA.git
cd  ./meshICA
pip install .
```