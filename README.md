<<<<<<< HEAD
# CanICA
Re-implementation of group ICA method for mesh-based 2D time-series data, based on the Varoquaux et al.'s 2010 *nilearn* CanICA implementation for 4D data:

  * http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html

  * https://www.ncbi.nlm.nih.gov/pubmed/20153834
  
Includes class to perform perform dual regression to generate subject-specific independent components and component-specific time series.

Currently, this implementation does not support spatial smoothing on the surface mesh.
=======
# meshICA
New implementations and reformatted implementations of group ICA methods for 2D mesh-based resting state MRI data.

  * **canica**: re-implementation of Canonical group ICA algorithm based on the Varoquaux et al.'s 2010 *nilearn* CanICA original implementation for 4D data:

    * http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
    * https://www.ncbi.nlm.nih.gov/pubmed/20153834
  
  
  * **migp**: implementation of Smith et al.'s 2014 Melodic's Iterative Group PCA algorithm for group ICA of very large datasets
  
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4289914/pdf/main.pdf
  
Includes class to perform perform dual regression to generate subject-specific independent components and component-specific time series.

This implementation does not currently support spatial smoothing on the surface mesh.
>>>>>>> 60265cb8b06c08216b927d736d6ec541317336a1
