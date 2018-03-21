# CanICA
New implementations and reformatted implementations of group ICA methods for 2D mesh-based resting state MRI data.

  * canica: re-implementation of group ICA Canonical ICA algorithm based on the Varoquaux et al.'s 2010 *nilearn* CanICA original implementation for 4D data:

   * http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
   * https://www.ncbi.nlm.nih.gov/pubmed/20153834
  
  * migp: implementation of Melodic's Iterative Group PCA algorithm for group ICA of very large datasets
  
Includes class to perform perform dual regression to generate subject-specific independent components and component-specific time series.

Currently, this implementation does not support spatial smoothing on the surface mesh.
