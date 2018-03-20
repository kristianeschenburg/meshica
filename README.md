# CanICA
Re-implementation of group ICA method for mesh-based 2D time-series data, based on the Varoquaux et al.'s 2010 *nilearn* CanICA implementation for 4D data:

  * http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html

  * https://www.ncbi.nlm.nih.gov/pubmed/20153834
  
Includes class to perform perform dual regression to generate subject-specific independent components and component-specific time series.

Currently, this implementation does not support spatial smoothing on the surface mesh.
