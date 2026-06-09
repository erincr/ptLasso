# Get the support for the overall model

Get the indices of nonzero coefficients from the overall model in a
fitted ptLasso or cv.ptLasso object, excluding the intercept.

## Usage

``` r
get.overall.support(fit, s = "lambda.min", gamma = "gamma.min")
```

## Arguments

- fit:

  fitted `"ptLasso"` or `"cv.ptLasso"` object.

- s:

  the choice of lambda to use. May be "lambda.min", "lambda.1se" or a
  numeric value. Default is "lambda.min".

- gamma:

  for use only when 'relax = TRUE' was specified during training. The
  choice of 'gamma' to use. May be "gamma.min" or "gamma.1se". Default
  is "gamma.min".

## Value

This returns a vector containing the indices of nonzero coefficients
(excluding the intercept).

## See also

`ptLasso`, `cv.ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
# Train data
set.seed(1234)
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group;

fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")

get.overall.support(fit, s="lambda.min") 
#>  [1]  1  2  4  5  6  7  8  9 10 11 12 13 26 29 31 36 40 45 46 58 59
get.overall.support(fit, s="lambda.1se") 
#> [1] 4 5 7

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")

get.overall.support(cvfit, s="lambda.min") 
#>  [1]  1  2  4  5  6  7 10 11 31 45 46 58 59
get.overall.support(cvfit, s="lambda.1se") 
#> [1] 4 5 7
 
```
