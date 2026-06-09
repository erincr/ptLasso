# Input grouped data

``` r

require(ptLasso)
#> Loading required package: ptLasso
#> Loading required package: ggplot2
#> Loading required package: glmnet
#> Loading required package: Matrix
#> Loaded glmnet 5.0
#> Loading required package: gridExtra
```

## Base case: input grouped data with a binomial outcome

In the Quick Start, we applied `ptLasso` to data with a continuous
response. Here, we’ll use data with a binary outcome. This creates a
dataset with $`k = 3`$ groups (each with $`100`$ observations), 5 shared
coefficients, and 5 coefficients specific to each group.

``` r

set.seed(1234)

out = binomial.example.data()
x = out$x; y = out$y; groups = out$groups

outtest = binomial.example.data()
xtest = outtest$x; ytest = outtest$y; groupstest = outtest$groups
```

We can fit and predict as before. By default, `predict.ptLasso` will
compute and return the *deviance* on the test set.

``` r

fit = ptLasso(x, y, groups, alpha = 0.5, family = "binomial")

predict(fit, xtest, groupstest, ytest = ytest)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest) 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (Deviance):
#> 
#>            allGroups  mean wtdMean group_1 group_2 group_3
#> Overall        1.359 1.359   1.359   1.334   1.321   1.421
#> Pretrain       1.279 1.279   1.279   1.272   1.169   1.397
#> Individual     1.283 1.283   1.283   1.265   1.186   1.399
#> 
#> Support size:
#>                                        
#> Overall    7                           
#> Pretrain   12 (3 common + 9 individual)
#> Individual 20
```

We could instead compute the AUC by specifying the `type.measure` in the
call to `ptLasso`. Note: `type.measure` is specified during model
fitting and not prediction because it is used in each call to
`cv.glmnet`.

``` r

fit = ptLasso(x, y, groups, alpha = 0.5, family = "binomial", 
              type.measure = "auc")

predict(fit, xtest, groupstest, ytest = ytest)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest) 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (AUC):
#> 
#>            allGroups   mean wtdMean group_1 group_2 group_3
#> Overall       0.6026 0.6039  0.6039  0.6161  0.6877  0.5080
#> Pretrain      0.6407 0.6524  0.6524  0.6936  0.7447  0.5190
#> Individual    0.6442 0.6618  0.6618  0.6936  0.7732  0.5186
#> 
#> Support size:
#>                                         
#> Overall    15                           
#> Pretrain   39 (3 common + 36 individual)
#> Individual 40
```

To fit the overall and individual models, we can use elasticnet instead
of lasso by defining the parameter `en.alpha` (as in `glmnet` and
described in the section “Fitting elasticnet or ridge models”).

``` r

fit = ptLasso(x, y, groups, alpha = 0.5, family = "binomial", 
              type.measure = "auc", 
              en.alpha = .5)
predict(fit, xtest, groupstest, ytest = ytest)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest) 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (AUC):
#> 
#>            allGroups   mean wtdMean group_1 group_2 group_3
#> Overall       0.6041 0.6018  0.6018  0.5928  0.6704  0.5422
#> Pretrain      0.6270 0.6547  0.6547  0.6781  0.7720  0.5141
#> Individual    0.6387 0.6598  0.6598  0.6756  0.7820  0.5218
#> 
#> Support size:
#>                                         
#> Overall    3                            
#> Pretrain   39 (3 common + 36 individual)
#> Individual 36
```

Using cross validation is the same as in the Gaussian case:

``` r

##################################################
# Fit:
##################################################
fit = cv.ptLasso(x, y, groups, family = "binomial", type.measure = "auc")

##################################################
# Predict with a common alpha for all groups:
##################################################
predict(fit, xtest, groupstest, ytest = ytest)

##################################################
# Predict with a different alpha for each group:
##################################################
predict(fit, xtest, groupstest, ytest = ytest, alphatype = "varying")
```

## Base case: input grouped survival data

``` r

require(survival)
#> Loading required package: survival
```

Now, we will simulate survival times with 3 groups; the three groups
have overlapping support, with 5 shared features and each has 5
individual features. To compute survival time, we start by computing
$`\text{survival} = X \beta + \epsilon`$, where $`\beta`$ is specific to
each group and $`\epsilon`$ is noise. Because survival times must be
positive, we modify this to be
$`\text{survival} = \text{survival} + 1.1 * \text{abs}(\text{min}(\text{survival}))`$.

``` r

set.seed(1234)

n = 600; ntrain = 300
p = 50
     
x = matrix(rnorm(n*p), n, p)
beta1 = c(rnorm(5), rep(0, p-5))

beta2 = runif(p) * beta1 # Shared support
beta2 = beta2 + c(rep(0, 5), rnorm(5), rep(0, p-10)) # Individual features

beta3 = runif(p) * beta1 # Shared support
beta3 = beta3 + c(rep(0, 10), rnorm(5), rep(0, p-15)) # Individual features

# Randomly split into groups
groups = sample(1:3, n, replace = TRUE)

# Compute survival times:
survival = x %*% beta1
survival[groups == 2] = x[groups == 2, ] %*% beta2
survival[groups == 3] = x[groups == 3, ] %*% beta3
survival = survival + rnorm(n)
survival = survival + 1.1 * abs(min(survival))

# Censoring times from a random uniform distribution:
censoring = runif(n, min = 1, max = 10)

# Did we observe surivival or censoring?
y = Surv(pmin(survival, censoring), survival <= censoring)

# Split into train and test:
xtest = x[-(1:300), ]
ytest = y[-(1:300), ]
groupstest = groups[-(1:300)]

x = x[1:300, ]
y = y[1:300, ]
groups = groups[1:300]
```

Training with `ptLasso` is much the same as it was for the continuous
and binomial cases; the only difference is that we specify
`family = "cox"`. By default, `ptLasso` uses the partial likelihood for
model selection. We could instead use the C index.

``` r

############################################################
# Default -- use partial likelihood as the type.measure:
############################################################
fit = ptLasso(x, y, groups, alpha = 0.5, family = "cox")
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -67); Convergence for 67th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
predict(fit, xtest, groupstest, ytest = ytest)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest) 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (Deviance):
#> 
#>            allGroups  mean wtdMean group_1 group_2 group_3
#> Overall        381.2 87.60   89.35   99.49  106.53   56.79
#> Pretrain       374.5 85.36   86.22   92.56   94.35   69.17
#> Individual     424.9 99.03   99.51  111.57  101.84   83.68
#> 
#> Support size:
#>                                         
#> Overall    10                           
#> Pretrain   15 (3 common + 12 individual)
#> Individual 24

############################################################
# Alternatively -- use the C index:
############################################################
fit = ptLasso(x, y, groups, alpha = 0.5, family = "cox", type.measure = "C")
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -51); Convergence for 51th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -54); Convergence for 54th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -54); Convergence for 54th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -60); Convergence for 60th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -57); Convergence for 57th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: from glmnet C++ code (error code -54); Convergence for 54th lambda
#> value not reached after maxit=100000 iterations; solutions for larger lambdas
#> returned
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
#> Warning: Starting in glmnet 5.1, the default Cox tie-handling method will
#> change from 'breslow' to 'efron' (matching survival::coxph). To silence this
#> message and lock in the v5.0 default, pass cox.ties = 'breslow' explicitly. To
#> preview the v5.1 behavior, pass cox.ties = 'efron'.
predict(fit, xtest, groupstest, ytest = ytest)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest) 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (C-index):
#> 
#>            allGroups   mean wtdMean group_1 group_2 group_3
#> Overall       0.8545 0.8673  0.8608  0.9139  0.7746  0.9133
#> Pretrain      0.8289 0.8376  0.8375  0.9165  0.8161  0.7802
#> Individual    0.7932 0.7995  0.8018  0.9075  0.8007  0.6904
#> 
#> Support size:
#>                                         
#> Overall    6                            
#> Pretrain   36 (4 common + 32 individual)
#> Individual 37
```

The call to `cv.ptLasso` is again much the same; we only need to specify
`family` (“cox”) and `type.measure` (if we want to use the C index
instead of the partial likelihood).

``` r

##################################################
# Fit:
##################################################
fit = cv.ptLasso(x, y, groups, family = "cox", type.measure = "C")

##################################################
# Predict with a common alpha for all groups:
##################################################
predict(fit, xtest, groupstest, ytest = ytest)

##################################################
# Predict with a different alpha for each group:
##################################################
predict(fit, xtest, groupstest, ytest = ytest, alphatype = "varying")
```
