# Pretraining and the Lasso

This package fits pretrained generalized linear models for (1) data with grouped observations and (2) data without grouped observations, but with multinomial responses. Details of this method may be found in Craig et al. ([2024](#ref-ptlasso)).

All model fitting in this package is done with `cv.glmnet`, and our syntax closely follows that of the `glmnet` package ([2010](#ref-glmnet)).

# References

<div id="refs" class="references">

<div id="ref-ptlasso">

Craig, Erin, Mert Pilanci, Thomas Le Menestrel, Balasubramanian Narasimhan, Manuel Rivas, Roozbeh Dehghannasiri, Julia Salzman, Jonathan Taylor, and Robert Tibshirani. "Pretraining and the Lasso." arXiv preprint arXiv:2401.12911 (2024).

<div id="ref-glmnet">

Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. 2010. “Regularization Paths for Generalized Linear Models via Coordinate Descent.” *Journal of Statistical Software, Articles* 33 (1): 1–22. <https://doi.org/10.18637/jss.v033.i01>.

</div>