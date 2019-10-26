# Mutual-Information-KDE
Computation of Mutual Information based on Kernel Density Estimation. Implemented in C++ using Eigen.

In order to use the code, compile it with g++ including eigen. No other dependecies needed.
Two files are required:
  - Input M x 1
  - Feature M x N

The Input file act as reference, against which the N columns of the Feature file will be used to compute the Mutual Information.
The Input and Feature file must have the same number of rows.

For now only 1D input are supported.

The program give as output also the three entropies H(Input), H(Feature), and H(Input, Feature). They are linked to the mutual information by the following:
I(Input|Feature) = H(Feature) + H(Input) - H(Input, Feature);

Relevant references:
- [Formula for estimating MI](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=256F92F8B18CEF221816BF21BED2AA2A?doi=10.1.1.713.5827&rep=rep1&type=pdf)
- [Rule for 1D Kernel Width](https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator)
- [Rule for 2D Kernel Width](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb)
- [Other reference for single/joint pdf estimation](https://pdfs.semanticscholar.org/29ac/cc5fabeabd9a567daaca379bb4073a4a2be4.pdf)
