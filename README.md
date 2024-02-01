## Welcome!
This README is very much a work-in-progress. Thank you for your patience!

## Installation 
### Dependencies
To use this package, you will need the following: `glmnet`, `Matrix`, `stats`, `ggplot2` and `gridExtra`.


### Installation
Installation is a little harder while this is a private repo:

```
credentials::set_github_pat() (enter your PAT into the popup)
remotes::install_github("erincr/ptLasso")
```

Alternatively, you can download or clone this repo, cd into the director containing it, and use: 

```R CMD build ptLasso``

followed by

```R CMD INSTALL ptLasso_1.0.tar.gz```.

## Examples

Given covariates ```x``` (nobs x nvars), outcome ```y``` (nobs), and group IDs ```groups``` (nobs, with group IDs from 1 to n_groups), you can call:

```
fit=ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",type.measure="mse")
```

To predict with data ```xtest```, ```groupstest``` and optionally ```ytest```:
```
pred=predict.ptLasso(fit,xtest,groupstest=groupstest, ytest=ytest)
```

The above assumes you have chosen a single value of ```alpha```. To tune this with CV, use:
```
cvfit=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
```

You can plot this:
```
plot(cvfit)
```

And you can predict using the single alpha that optimizes the overall performance:
```
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="fixed")
```

or using a vector of alphas - one that optimizes the CV performance for each group:
```
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
```


