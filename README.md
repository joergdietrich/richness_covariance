This Python module computes the covariance contribution of the finite richness binnin in the redMaPPer mass calibration.

It requires https://github.com/joergdietrich/NFW for computation of the NFW Delta Sigma profiles, [Diemer's colossus](https://bitbucket.org/bdiemer/colossus) for the DK15 M-c relation, and the [tqdm progress meter](https://pypi.python.org/pypi/tqdm)

Outputs are ndarray covariance matrices in units of (Msun/Mpc^2)^2. All quantities are physical without factors of h. They are saved as Python pickles in the directory `output`.
