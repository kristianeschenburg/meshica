# meshICA
New implementations and reformatted implementations of group ICA methods for 2D mesh-based resting state MRI data.

  * **canica**: re-implementation of group ICA for 2D data based on the Varoquaux et al.'s 2010 CanonicalICA algorithm implementation in the *nilearn* package for 4D volumetric data:

    * http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
    * https://www.ncbi.nlm.nih.gov/pubmed/20153834
  
  
  * **migp** (in-progress): implementation of Smith et al.'s 2014 Melodic's Iterative Group PCA (MIGP) algorithm for group ICA of very large datasets
  
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4289914/pdf/main.pdf
  
Includes class to perform perform dual regression to generate subject-specific independent components and component-specific time series.

This implementation does not currently support spatial smoothing on the surface mesh.
