# Pretraining and the Lasso

This package fits pretrained generalized linear models for: (1) data
with grouped observations, (2) data without grouped observations, but
with multinomial responses, (3) data with multiple Gaussian responses
and (4) time series data (data with repeated measurements over time).

Documentation and examples are available as vignettes within this
package, and can be accessed through the “Articles” tab on this page.
The vignettes also include examples of pretraining for settings not yet
supported by this package, including conditional average treatment
effect estimation and unsupervised pretraining.

Details of pretraining may be found in Craig et
al. ([2026](#ref-ptlasso)).

All model fitting in this package is done with `cv.glmnet`, and our
syntax closely follows that of the `glmnet` package
([2010](#ref-glmnet)).

# Tutorials

## Tutorial 1: an introduction

An introduction to pretraining, and to the R package:

- video ([YouTube](https://www.youtube.com/watch?v=zIGc5Z2MaYM))
- slides
  ([.pdf](https://erincr.github.io/ptLasso/tutorials/ptLasso_tutorial_1.pdf))
- code
  ([.html](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part1.html),
  [.Rmd](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part1.Rmd))

## Tutorial 2: a deeper dive and more examples

More examples of pretraining and modeling:

- video ([YouTube](https://www.youtube.com/watch?v=97Kej1iomZ8))
- slides
  ([.pdf](https://erincr.github.io/ptLasso/tutorials/ptLasso_tutorial_2.pdf))
- code
  ([.html](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part2.html),
  [.Rmd](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part2.Rmd))

# Installation

To install this package, we recommend following [these
instructions](https://cran.r-project.org/web/packages/githubinstall/vignettes/githubinstall.html).

# Having trouble?

If you find a bug or have a feature request, please open a new issue.

# References

Craig et al. “Pretraining and the lasso.” *Journal of the Royal
Statistical Society Series B: Statistical Methodology* 88.1 (2026):
261-281.

Friedman, Hastie, and Tibshirani. 2010. “Regularization Paths for
Generalized Linear Models via Coordinate Descent.” *Journal of
Statistical Software, Articles* 33 (1): 1–22.
