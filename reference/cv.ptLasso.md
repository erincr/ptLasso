# Cross-validation for ptLasso

Cross-validation for `ptLasso`.

## Usage

``` r
cv.ptLasso(
  x,
  y,
  groups = NULL,
  alphalist = seq(0, 1, length = 11),
  family = c("default", "gaussian", "multinomial", "binomial", "cox"),
  use.case = c("inputGroups", "targetGroups", "multiresponse", "timeSeries"),
  type.measure = c("default", "mse", "mae", "auc", "deviance", "class", "C"),
  nfolds = 10,
  foldid = NULL,
  verbose = FALSE,
  fitoverall = NULL,
  fitind = NULL,
  s = "lambda.min",
  gamma = "gamma.min",
  alphahat.choice = "overall",
  group.intercepts = TRUE,
  ...
)
```

## Arguments

- x:

  `x` matrix as in `ptLasso`.

- y:

  `y` vector or matrix as in `ptLasso`.

- groups:

  A vector of length nobs indicating to which group each observation
  belongs. For data with k groups, groups should be coded as integers 1
  through k. Only for 'use.case = "inputGroups"'.

- alphalist:

  A vector of values of the pretraining hyperparameter alpha. Defaults
  to `seq(0, 1, length.out=11)`. This function will do pretraining for
  each choice of alpha in alphalist and return the CV performance for
  each alpha.

- family:

  Response type as in `ptLasso`.

- use.case:

  The type of grouping observed in the data. Can be one of
  "inputGroups", "targetGroups", "multiresponse" or "timeSeries".

- type.measure:

  Measure computed in `cv.glmnet`, as in `ptLasso`.

- nfolds:

  Number of folds for CV (default is 10). Although `nfolds`can be as
  large as the sample size (leave-one-out CV), it is not recommended for
  large datasets. Smallest value allowable is `nfolds = 3`.

- foldid:

  An optional vector of values between 1 and `nfolds` identifying what
  fold each observation is in. If supplied, `nfolds` can be missing.

- verbose:

  If `verbose=1`, print a statement showing which model is currently
  being fit.

- fitoverall:

  An optional cv.glmnet object specifying the overall model. This should
  have been trained on the full training data, with the argument keep =
  TRUE.

- fitind:

  An optional list of cv.glmnet objects specifying the individual
  models. These should have been trained on the training data, with the
  argumnet keep = TRUE.

- s:

  The choice of lambda to be used by all models when estimating the CV
  performance for each choice of alpha. Defaults to "lambda.min". May be
  "lambda.1se", or a numeric value. (Use caution when supplying a
  numeric value: the same lambda will be used for all models.)

- gamma:

  For use only when `relax = TRUE`. The choice of gamma to be used by
  all models when estimating the CV performance for each choice of
  alpha. Defaults to "gamma.min". May also be "gamma.1se".

- alphahat.choice:

  When choosing alphahat, we may prefer the best performance using all
  data (`alphahat.choice = "overall"`) or the best average performance
  across groups (`alphahat.choice = "mean"`). This is particularly
  useful when `type.measure` is "auc" or "C", because the average
  performance across groups is different than the performance with the
  full dataset. The default is "overall".

- group.intercepts:

  For 'use.case = "inputGroups"' only. If \`TRUE\`, fit the overall
  model with a separate intercept for each group. If \`FALSE\`, ignore
  the grouping and fit one overall intercept. Default is \`TRUE\`.

- ...:

  Additional arguments to be passed to the \`cv.glmnet\` function.
  Notable choices include `"trace.it"` and `"parallel"`. If
  `trace.it = TRUE`, then a progress bar is displayed for each call to
  `cv.glmnet`; useful for big models that take a long time to fit. If
  `parallel = TRUE`, use parallel `foreach` to fit each fold. Must
  register parallel before hand, such as `doMC` or others. Importantly,
  `"cv.ptLasso"` does not support the arguments `"intercept"`,
  `"offset"`, `"fit"` and `"check.args"`.

## Value

An object of class `"cv.ptLasso"`, which is a list with the ingredients
of the cross-validation fit.

- call:

  The call that produced this object.

- alphahat:

  Value of `alpha` that optimizes CV performance on all data.

- varying.alphahat:

  Vector of values of `alpha`, the kth of which optimizes performance
  for group k.

- alphalist:

  Vector of all alphas that were compared.

- errall:

  CV performance for the overall model.

- errpre:

  CV performance for the pretrained models (one for each `alpha` tried).

- errind:

  CV performance for the individual model.

- fit:

  List of `ptLasso` objects, one for each `alpha` tried.

- fitoverall:

  The fitted overall model used for the first stage of pretraining.

- fitoverall.lambda:

  The value of `lambda` used for the first stage of pretraining.

- fitind:

  A list containing one individual model for each group.

- use.case:

  The use case: "inputGroups" or "targetGroups".

- family:

  The family used.

- type.measure:

  The type.measure used.

## Details

This function runs `ptLasso` once for each requested choice of alpha,
and returns the cross validated performance.

## See also

[`ptLasso`](https://erincr.github.io/ptLasso/reference/ptLasso.md) and
[`plot.cv.ptLasso`](https://erincr.github.io/ptLasso/reference/plot.cv.ptLasso.md).

## Examples

``` r
# Getting started. First, we simulate data: we need covariates x, response y and group IDs.
set.seed(1234)
n = 80
p = 20
x = matrix(rnorm(n*p), n, p)
y = rnorm(n)
groups = sort(rep(1:5, n/5))

xtest = matrix(rnorm(n*p), n, p)
ytest = rnorm(n)
groupstest = sort(rep(1:5, n/5))

# Model fitting
cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", nfolds=3, 
                   type.measure = "mse")
cvfit
#> 
#> Call:  
#> cv.ptLasso(x = x, y = y, groups = groups, family = "gaussian",  
#>     type.measure = "mse", nfolds = 3, use.case = "inputGroups",  
#>     group.intercepts = TRUE) 
#> 
#> 
#> type.measure:  mse 
#> 
#> 
#>            alpha overall  mean wtdMean group_1 group_2 group_3 group_4 group_5
#> Overall            1.318 1.318   1.318  1.0289   1.761   1.447   1.519  0.8360
#> Pretrain     0.0   1.555 1.555   1.555  1.2253   1.932   1.635   1.870  1.1151
#> Pretrain     0.1   1.772 1.772   1.772  1.9005   2.454   1.665   1.808  1.0337
#> Pretrain     0.2   1.586 1.586   1.586  1.2015   1.946   1.973   1.803  1.0073
#> Pretrain     0.3   1.630 1.630   1.630  1.7202   1.939   1.502   1.770  1.2186
#> Pretrain     0.4   1.419 1.419   1.419  1.1071   1.895   1.634   1.378  1.0786
#> Pretrain     0.5   1.506 1.506   1.506  0.9974   1.527   1.842   2.219  0.9469
#> Pretrain     0.6   1.639 1.639   1.639  1.3574   2.295   1.654   1.578  1.3111
#> Pretrain     0.7   1.360 1.360   1.360  0.9112   1.505   1.690   1.493  1.2029
#> Pretrain     0.8   1.387 1.387   1.387  0.9263   1.642   1.556   1.960  0.8495
#> Pretrain     0.9   1.305 1.305   1.305  1.0910   1.354   1.546   1.419  1.1155
#> Pretrain     1.0   1.252 1.252   1.252  1.0629   1.162   1.526   1.387  1.1244
#> Individual         1.252 1.252   1.252  1.0629   1.162   1.526   1.387  1.1244
#> 
#> alphahat (fixed) = 1
#> alphahat (varying):
#> group_1 group_2 group_3 group_4 group_5 
#>     0.7     1.0     0.3     0.4     0.8 
plot(cvfit) # to see CV performance as a function of alpha 
predict(cvfit, xtest, groupstest, s="lambda.min") # to predict with held out data
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     s = "lambda.min") 
#> 
#> 
#> alpha =  1 
#> 
#> Support size:
#>                                       
#> Overall    0                          
#> Pretrain   1 (0 common + 1 individual)
#> Individual 1                          
predict(cvfit, xtest, groupstest, s="lambda.min", ytest=ytest) # to also measure performance
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, s = "lambda.min") 
#> 
#> 
#> alpha =  1 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2 group_3 group_4 group_5      r^2
#> Overall        1.223 1.223   1.131  0.7117  0.7803   1.809   1.682 -0.02313
#> Pretrain       1.223 1.223   1.131  0.7104  0.7803   1.809   1.682 -0.02292
#> Individual     1.223 1.223   1.131  0.7104  0.7803   1.809   1.682 -0.02292
#> 
#> Support size:
#>                                       
#> Overall    0                          
#> Pretrain   1 (0 common + 1 individual)
#> Individual 1                          

# By default, we used s = "lambda.min" to compute CV performance.
# We could instead use s = "lambda.1se":
cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", nfolds=3, 
                   type.measure = "mse", s = "lambda.1se")

# \donttest{
# We could have used the glmnet option relax = TRUE:
cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", nfolds=3, 
                   type.measure = "mse", relax = TRUE)
# And, as we did with lambda, we may want to specify the choice of gamma to compute CV performance:
cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", nfolds=3, 
                   type.measure = "mse", relax = TRUE, gamma = "gamma.1se")
# }
# Note that the first stage of pretraining uses "lambda.1se" and "gamma.1se" by default.
# This behavior can be modified by specifying overall.lambda and overall.gamma;
# see the documentation for ptLasso for more information.

# \donttest{
# Now, we are ready to simulate slightly more realistic data.
# This continuous outcome example has k = 5 groups, where each group has 200 observations.
# There are scommon = 10 features shared across all groups, and
# sindiv = 10 features unique to each group.
# n = 1000 and p = 120 (60 informative features and 60 noise features).
# The coefficients of the common features differ across groups (beta.common).
# In group 1, these coefficients are rep(1, 10); in group 2 they are rep(2, 10), etc.
# Each group has 10 unique features, the coefficients of which are all 3 (beta.indiv).
# The intercept in all groups is 0.
# The variable sigma = 20 indicates that we add noise to y according to 20 * rnorm(n). 
set.seed(1234)
k=5
class.sizes=rep(200, k)
scommon=10; sindiv=rep(10, k)
n=sum(class.sizes); p=2*(sum(sindiv) + scommon)
beta.common=3*(1:k); beta.indiv=rep(3, k)
intercepts=rep(0, k)
sigma=20
out = gaussian.example.data(k=k, class.sizes=class.sizes,
                            scommon=scommon, sindiv=sindiv,
                            n=n, p=p,
                            beta.common=beta.common, beta.indiv=beta.indiv,
                            intercepts=intercepts, sigma=20)
x = out$x; y=out$y; groups = out$group

outtest = gaussian.example.data(k=k, class.sizes=class.sizes,
                                scommon=scommon, sindiv=sindiv,
                                n=n, p=p,
                                beta.common=beta.common, beta.indiv=beta.indiv,
                                intercepts=intercepts, sigma=20)
xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
cvfit
#> 
#> Call:  
#> cv.ptLasso(x = x, y = y, groups = groups, family = "gaussian",  
#>     type.measure = "mse", use.case = "inputGroups", group.intercepts = TRUE) 
#> 
#> 
#> 
#> type.measure:  mse 
#> 
#> 
#>            alpha overall  mean wtdMean group_1 group_2 group_3 group_4 group_5
#> Overall            699.7 699.7   699.7   748.4   501.9   575.6   663.0  1009.9
#> Pretrain     0.0   518.6 518.6   518.6   470.1   471.5   547.0   540.7   563.7
#> Pretrain     0.1   506.0 506.0   506.0   429.7   452.1   538.7   551.1   558.3
#> Pretrain     0.2   495.3 495.3   495.3   393.6   460.6   565.5   530.9   526.1
#> Pretrain     0.3   490.4 490.4   490.4   390.4   436.5   546.3   511.6   567.4
#> Pretrain     0.4   487.5 487.5   487.5   383.7   438.8   545.6   509.4   560.3
#> Pretrain     0.5   481.2 481.2   481.2   364.9   429.7   548.5   513.4   549.7
#> Pretrain     0.6   504.1 504.1   504.1   393.1   460.0   586.4   531.9   549.0
#> Pretrain     0.7   511.5 511.5   511.5   393.2   462.7   584.3   492.9   624.3
#> Pretrain     0.8   509.1 509.1   509.1   382.4   496.2   597.9   503.4   565.6
#> Pretrain     0.9   501.5 501.5   501.5   404.0   481.6   581.9   488.3   552.0
#> Pretrain     1.0   517.1 517.1   517.1   409.1   488.9   612.7   484.7   590.1
#> Individual         517.1 517.1   517.1   409.1   488.9   612.7   484.7   590.1
#> 
#> alphahat (fixed) = 0.5
#> alphahat (varying):
#> group_1 group_2 group_3 group_4 group_5 
#>     0.5     0.5     0.1     1.0     0.2 
# plot(cvfit) # to see CV performance as a function of alpha 
predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, s = "lambda.min") 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2 group_3 group_4 group_5    r^2
#> Overall        755.7 755.7   836.0   554.9   565.4   777.9  1044.0 0.5371
#> Pretrain       500.2 500.2   539.0   443.8   553.5   502.5   462.4 0.6936
#> Individual     532.8 532.8   584.1   443.2   567.2   550.5   518.9 0.6736
#> 
#> Support size:
#>                                          
#> Overall    64                            
#> Pretrain   92 (21 common + 71 individual)
#> Individual 109                           
# }

# \donttest{
# Now, we repeat with a binomial outcome.
# This example has k = 3 groups, where each group has 100 observations.
# There are scommon = 5 features shared across all groups, and
# sindiv = 5 features unique to each group.
# n = 300 and p = 40 (20 informative features and 20 noise features).
# The coefficients of the common features differ across groups (beta.common),
# as do the coefficients specific to each group (beta.indiv).
set.seed(1234)
k=3
class.sizes=rep(100, k)
scommon=5; sindiv=rep(5, k)
n=sum(class.sizes); p=2*(sum(sindiv) + scommon)
beta.common=list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3))
beta.indiv = lapply(1:k, function(i)  0.9 * beta.common[[i]])

out = binomial.example.data(k=k, class.sizes=class.sizes,
                            scommon=scommon, sindiv=sindiv,
                            n=n, p=p,
                            beta.common=beta.common, beta.indiv=beta.indiv)
x = out$x; y=out$y; groups = out$group

outtest = binomial.example.data(k=k, class.sizes=class.sizes,
                                scommon=scommon, sindiv=sindiv,
                                n=n, p=p,
                                beta.common=beta.common, beta.indiv=beta.indiv)
xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups

cvfit = cv.ptLasso(x, y, groups = groups, family = "binomial",
                   type.measure = "auc", nfolds=3, verbose=TRUE, 
                   alpha = c(0, .5, 1),
                   alphahat.choice="mean")
#> 
#> alpha= 0
#> Fitting overall model
#> Fitting individual models
#>  Fitting individual model 1 / 3
#>  Fitting individual model 2 / 3
#>  Fitting individual model 3 / 3
#> Fitting pretrained lasso models
#>  Fitting pretrained model 1 / 3
#>  Fitting pretrained model 2 / 3
#>  Fitting pretrained model 3 / 3
#> 
#> alpha= 0.5
#> Fitting pretrained lasso models
#>  Fitting pretrained model 1 / 3
#>  Fitting pretrained model 2 / 3
#>  Fitting pretrained model 3 / 3
#> 
#> alpha= 1
#> Fitting pretrained lasso models
cvfit
#> 
#> Call:  
#> cv.ptLasso(x = x, y = y, groups = groups, alphalist = c(0, 0.5,  
#>     1), family = "binomial", type.measure = "auc", nfolds = 3,  
#>     verbose = TRUE, alphahat.choice = "mean", use.case = "inputGroups",  
#>     group.intercepts = TRUE) 
#> 
#> 
#> type.measure:  auc 
#> 
#> 
#>            alpha overall   mean wtdMean group_1 group_2 group_3
#> Overall           0.5715 0.5717  0.5717  0.6933  0.5994  0.4223
#> Pretrain     0.0  0.6272 0.6242  0.6242  0.8001  0.6819  0.3905
#> Pretrain     0.5  0.7284 0.7623  0.7623  0.8472  0.7636  0.6762
#> Pretrain     1.0  0.5882 0.6894  0.6894  0.7896  0.7112  0.5673
#> Individual        0.5882 0.6894  0.6894  0.7896  0.7112  0.5673
#> 
#> alphahat (fixed) = 0.5
#> alphahat (varying):
#> group_1 group_2 group_3 
#>     0.5     0.5     0.5 
# plot(cvfit) # to see CV performance as a function of alpha 
predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.1se")
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, s = "lambda.1se") 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (AUC):
#> 
#>            allGroups   mean wtdMean group_1 group_2 group_3
#> Overall       0.6070 0.6072  0.6072  0.5985  0.6764  0.5467
#> Pretrain      0.6391 0.6440  0.6440  0.6720  0.7166  0.5435
#> Individual    0.6762 0.6435  0.6435  0.6663  0.7644  0.5000
#> 
#> Support size:
#>                                         
#> Overall    3                            
#> Pretrain   21 (3 common + 18 individual)
#> Individual 7                            
# }

if (FALSE) { # \dontrun{
### Model fitting with parallel = TRUE
require(doMC)
registerDoMC(cores = 4)
cvfit = cv.ptLasso(x, y, groups = groups, family = "binomial",
                   type.measure = "auc", parallel=TRUE)
} # }
# \donttest{
# Multiresponse pretraining
# Now let's consider the case of a multiresponse outcome. We'll start by simulating data:
set.seed(1234)
n = 1000; ntrain = 500;
p = 500
sigma = 2
     
x = matrix(rnorm(n*p), n, p)
beta1 = c(rep(1, 5), rep(0.5, 5), rep(0, p - 10))
beta2 = c(rep(1, 5), rep(0, 5), rep(0.5, 5), rep(0, p - 15))

mu = cbind(x %*% beta1, x %*% beta2)
y  = cbind(mu[, 1] + sigma * rnorm(n), 
           mu[, 2] + sigma * rnorm(n))
cat("SNR for the two tasks:", round(diag(var(mu)/var(y-mu)), 2), fill=TRUE)
#> SNR for the two tasks: 1.6 1.44
cat("Correlation between two tasks:", cor(y[, 1], y[, 2]), fill=TRUE)
#> Correlation between two tasks: 0.5164748

xtest = x[-(1:ntrain), ]
ytest = y[-(1:ntrain), ]

x = x[1:ntrain, ]
y = y[1:ntrain, ]

# Now, we can fit a ptLasso model:
fit = cv.ptLasso(x, y, type.measure = "mse", use.case = "multiresponse")
plot(fit) # to see the cv curve.
predict(fit, xtest) # to predict with new data
#> 
#> Call:  
#> predict.cv.ptLasso(object = fit, xtest = xtest) 
#> 
#> 
#> 
#> alpha =  0.2 
#> 
#> Support size:
#>                                         
#> Overall    57                           
#> Pretrain   23 (19 common + 4 individual)
#> Individual 80                           
predict(fit, xtest, ytest=ytest) # if ytest is included, we also measure performance
#> 
#> Call:  
#> predict.cv.ptLasso(object = fit, xtest = xtest, ytest = ytest) 
#> 
#> 
#> 
#> alpha =  0.2 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean response_1 response_2
#> Overall        9.394 4.697      4.227      5.168
#> Pretrain       8.907 4.453      4.186      4.721
#> Individual     9.465 4.733      4.243      5.222
#> 
#> Support size:
#>                                         
#> Overall    57                           
#> Pretrain   23 (19 common + 4 individual)
#> Individual 80                           
# By default, we used s = "lambda.min" to compute CV performance.
# We could instead use s = "lambda.1se":
cvfit = cv.ptLasso(x, y, type.measure = "mse", s = "lambda.1se", use.case = "multiresponse")

# We could also use the glmnet option relax = TRUE:
cvfit = cv.ptLasso(x, y, type.measure = "mse", relax = TRUE, use.case = "multiresponse")
# And, as we did with lambda, we may want to specify the choice of gamma to compute CV performance:
cvfit = cv.ptLasso(x, y, type.measure = "mse", relax = TRUE, gamma = "gamma.1se",
                   use.case = "multiresponse")
# }

# \donttest{
# Time series pretraining
# Now suppose we have time series data with a binomial outcome measured at 3 different time points.
set.seed(1234)
n = 600; ntrain = 300; p = 50
x = matrix(rnorm(n*p), n, p)

beta1 = c(rep(0.25, 10), rep(0, p-10))
beta2 = beta1 + c(rep(0.1, 10), runif(5, min = -0.25, max = 0), rep(0, p-15))
beta3 = beta1 + c(rep(0.2, 10), runif(5, min = -0.25, max = 0),
                  runif(5, min = 0, max = 0.1), rep(0, p-20))

y1 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta1)))
y2 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta2)))
y3 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta3)))
y = cbind(y1, y2, y3)

xtest = x[-(1:ntrain), ]
ytest = y[-(1:ntrain), ]

x = x[1:ntrain, ]
y = y[1:ntrain, ]

cvfit =  cv.ptLasso(x, y, use.case="timeSeries", family="binomial",
                    type.measure="auc")
plot(cvfit, plot.alphahat = TRUE)
predict(cvfit, xtest, ytest=ytest)
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, ytest = ytest) 
#> 
#> 
#> 
#> alpha =  0.8 
#> 
#> Performance (AUC):
#> 
#>              mean response_1 response_2 response_3
#> Pretrain   0.7093     0.6731     0.7165     0.7383
#> Individual 0.7042     0.6731     0.7116     0.7279
#> 
#> Support size:
#>                                          
#> Pretrain   39 (21 common + 18 individual)
#> Individual 39                            

# The glmnet option relax = TRUE:
cvfit = cv.ptLasso(x, y, type.measure = "auc", family = "binomial", relax = TRUE,
                   use.case = "timeSeries")
# }
```
