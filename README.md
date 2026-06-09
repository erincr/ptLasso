# Pretraining and the Lasso

Pretraining is a popular and powerful paradigm in machine learning. For example, suppose we have a modest-sized dataset of images of cats and dogs, and we want to fit a deep neural network to classify them. With pretraining, we start with a neural network trained on a large corpus of images, *consisting of not just cats and dogs but hundreds of other image types.* Then we fix all of the network weights except for the top layer(s) and train (or "fine tune") those weights on our dataset. This often results in dramatically better performance than the network trained solely on our smaller dataset.

Why should this work? By training our neural network first with a large, broad set of images, we learn general features like edges, textures, fur, eyes and so on. Then, during the fine tuning stage, we learn how to use these features to distinguish between cats and dogs.

**Can pretraining help the lasso? Yes!**

Here we present a framework for the lasso in which an overall model is fit to a large set of data, and then fine-tuned to a specific task. This latter dataset can be a subset of the original dataset, but does not need to be. Pretraining the lasso has a wide variety of applications, including stratified models, multinomial targets, multi-response models, conditional average treatment estimation and even gradient boosting.

This package fits pretrained generalized linear models for: (1) data with grouped observations, (2) data without grouped observations, but with multinomial responses, (3) data with multiple Gaussian responses and (4) time series data (data with repeated measurements over time).

Documentation and examples are available as vignettes within this package, and can be accessed through the "Articles" tab on this page. The vignettes also include examples of pretraining for settings not yet supported by this package, including conditional average treatment effect estimation and unsupervised pretraining.

Details of pretraining may be found in Craig et al. ([2026](#ref-ptlasso)).

All model fitting in this package is done with `cv.glmnet`, and our syntax closely follows that of the `glmnet` package ([2010](#ref-glmnet)).

# Tutorials

## Tutorial 1: an introduction

An introduction to pretraining, and to the R package:

- video ([YouTube](https://www.youtube.com/watch?v=zIGc5Z2MaYM))
- slides ([.pdf](https://erincr.github.io/ptLasso/tutorials/ptLasso_tutorial_1.pdf))
- code ([.html](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part1.html), [.Rmd](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part1.Rmd))

## Tutorial 2: a deeper dive and more examples

More examples of pretraining and modeling:

- video ([YouTube](https://www.youtube.com/watch?v=97Kej1iomZ8))
- slides ([.pdf](https://erincr.github.io/ptLasso/tutorials/ptLasso_tutorial_2.pdf))
- code ([.html](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part2.html), [.Rmd](https://erincr.github.io/ptLasso/tutorials/ptLasso_Part2.Rmd))

# Installation

To install this package, we recommend following [these instructions](https://cran.r-project.org/web/packages/githubinstall/vignettes/githubinstall.html).

# Having trouble?

If you find a bug or have a feature request, please open a new issue.

# References

<a id="ref-ptlasso"></a>Craig et al. "Pretraining and the lasso." *Journal of the Royal Statistical Society Series B: Statistical Methodology* 88.1 (2026): 261-281.

<a id="ref-glmnet"></a>Friedman, Hastie, and Tibshirani. 2010. "Regularization Paths for Generalized Linear Models via Coordinate Descent." *Journal of Statistical Software, Articles* 33 (1): 1–22.