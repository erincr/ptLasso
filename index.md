# Pretraining and the Lasso

This package fits pretrained generalized linear models for: 
(1) data with grouped observations,
(2) data without grouped observations, but with multinomial responses,
(3) data with multiple Gaussian responses and
(4) time series data (data with repeated measurements over time).

Documentation and examples are available as vignettes within this package, and can be accessed through the "Articles" tab on this page.
The vignettes also include examples of pretraining for settings not yet supported by this package, including conditional average treatment effect estimation and unsupervised pretraining.

Details of pretraining may be found in Craig et al. ([2024](#ref-ptlasso)).

All model fitting in this package is done with `cv.glmnet`, and our syntax closely follows that of the `glmnet` package ([2010](#ref-glmnet)).

# Tutorials 

For introductory YouTube tutorials and R Markdown examples, please visit the [website](https://erincraig.me/lasso-pretraining) for lasso pretraining.

# Installation
To install this package, we recommend following [these instructions](https://cran.r-project.org/web/packages/githubinstall/vignettes/githubinstall.html).

# Having trouble?
If you find a bug or have a feature request, please open a new issue.

# References

<div id="refs" class="references">

<div id="ref-ptlasso">

Craig, Erin, Mert Pilanci, Thomas Le Menestrel, Balasubramanian Narasimhan, Manuel Rivas, Roozbeh Dehghannasiri, Julia Salzman, Jonathan Taylor, and Robert Tibshirani. "Pretraining and the Lasso." arXiv preprint [arXiv:2401.12911](https://arxiv.org/abs/2401.12911) (2024).

</div>

<div id="ref-glmnet">

Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. 2010. “Regularization Paths for Generalized Linear Models via Coordinate Descent.” *Journal of Statistical Software, Articles* 33 (1): 1–22. <https://doi.org/10.18637/jss.v033.i01>.

</div>

</div>
