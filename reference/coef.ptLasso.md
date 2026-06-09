# Get the coefficients from a fitted ptLasso model.

Get the coefficients from a fitted ptLasso model.

## Usage

``` r
# S3 method for class 'ptLasso'
coef(object, model = c("all", "individual", "overall", "pretrain"), ...)
```

## Arguments

- object:

  fitted `"ptLasso"` object.

- model:

  string indicating which coefficients to retrieve. Must be one of
  "all", "individual", "overall" or "pretrain".

- ...:

  other arguments to be passed to the `"coef"` function. May be e.g.
  `s = "lambda.min"`.

## Value

Model coefficients. If `model = "overall"`, this function returns the
output of `coef`. If `model` is "individual" or "pretrain", this
function returns a list containing the results of `coef` for each
group-specific model. If `model = "all"`, this returns a list containing
all (overall, individual and pretrain) coefficients.

## See also

`ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
# Train data
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group;

fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
# Get all model coefficients.
names(coef(fit))
#> [1] "individual" "pretrain"   "overall"   

coef(fit, model = "overall") # Overall model only
#> 125 x 1 sparse Matrix of class "dgCMatrix"
#>             lambda.1se
#> (Intercept)  -2.333542
#> groups2      -1.176982
#> groups3       2.184525
#> groups4       2.632415
#> groups5      -5.340553
#>               6.739073
#>               7.657666
#>               6.096106
#>               6.309123
#>               8.605630
#>               7.535845
#>               7.214832
#>               7.635445
#>               7.697436
#>               7.149646
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
#>               .       
length(coef(fit, model = "individual")) # List of coefficients for each group model
#> [1] 5
length(coef(fit, model = "pretrain")) # List of coefficients for each group model
#> [1] 5
```
