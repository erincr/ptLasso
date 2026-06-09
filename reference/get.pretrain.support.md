# Get the support for pretrained models

Get the indices of nonzero coefficients from the pretrained models in a
fitted ptLasso or cv.ptLasso object, excluding the intercept.

## Usage

``` r
get.pretrain.support(
  fit,
  s = "lambda.min",
  gamma = "gamma.min",
  commonOnly = FALSE,
  includeOverall = TRUE,
  groups = 1:length(fit$fitind)
)
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

- commonOnly:

  whether to return the features that are chosen by more than half of
  the group- or response-specific models (TRUE) or the features that are
  chosen by any of the group-specific models (FALSE). Default is FALSE.

- includeOverall:

  whether to return the features that are chosen by the overall model
  and not the group-specific models (TRUE) or the features that are
  chosen by the overall model or the group-specific models (FALSE).
  Default is TRUE. Not used when 'use.case = "timeSeries"'.

- groups:

  which groups or responses to include when computing the support.
  Default is to include all groups/responses.

## Value

If a ptLasso object is supplied, this returns a vector containing the
indices of nonzero coefficients (excluding the intercept). If a
cv.ptLasso object is supplied, this returns a list of results - one for
each value of alpha.

## See also

`ptLasso`, `cv.ptLasso`.

## Examples

``` r
# Train data
set.seed(1234)
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group;

fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")

get.pretrain.support(fit) 
#>  [1]  1  4  5  6  7 11 40 45 46 58

# only return features common to all groups 
get.pretrain.support(fit, commonOnly = TRUE) 
#> [1] 4 5 7

# group 1 only, don't include the overall model support
get.pretrain.support(fit, groups = 1, includeOverall = FALSE) 
#> integer(0)

# group 1 only, include the overall model support
get.pretrain.support(fit, groups = 1, includeOverall = TRUE) 
#> [1] 4 5 7

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")

get.pretrain.support(cvfit)
#> [[1]]
#> [1] 4 5 7
#> 
#> [[2]]
#> [1]  4  5  6  7 11 45 58
#> 
#> [[3]]
#>  [1]  1  2  4  5  6  7  8 11 26 40 45 46 58
#> 
#> [[4]]
#>  [1]  1  2  4  5  6  7 11 26 40 45 46 58
#> 
#> [[5]]
#>  [1]  1  4  5  6  7 11 40 45 46 58
#> 
#> [[6]]
#>  [1]  1  4  5  6  7 11 40 45 46 58
#> 
#> [[7]]
#>  [1]  1  2  4  5  6  7  8 11 26 40 45 46 58
#> 
#> [[8]]
#>  [1]  1  2  4  5  6  7  8 11 26 40 45 46 58
#> 
#> [[9]]
#>  [1]  1  2  4  5  6  7  8 11 26 40 45 46 58
#> 
#> [[10]]
#>  [1]  1  2  4  5  6  7  8  9 10 11 13 22 25 26 28 31 32 34 40 42 45 46 51 58
#> 
#> [[11]]
#>  [1]  1  2  4  5  6  7  8  9 10 11 12 13 22 25 26 28 31 32 34 40 41 42 45 46 51
#> [26] 54 58
#> 
get.pretrain.support(cvfit, groups = 1) 
#> [[1]]
#> [1] 4 5 7
#> 
#> [[2]]
#> [1] 4 5 7
#> 
#> [[3]]
#> [1] 4 5 7
#> 
#> [[4]]
#> [1] 4 5 7
#> 
#> [[5]]
#> [1] 4 5 7
#> 
#> [[6]]
#> [1] 4 5 7
#> 
#> [[7]]
#> [1] 4 5 7
#> 
#> [[8]]
#> [1] 4 5 7
#> 
#> [[9]]
#> [1] 4 5 7
#> 
#> [[10]]
#> [1] 4 5 7
#> 
#> [[11]]
#> [1]  6 32 54
#> 
 
```
