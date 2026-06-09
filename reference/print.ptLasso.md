# Print the ptLasso object.

Print the ptLasso object.

## Usage

``` r
# S3 method for class 'ptLasso'
print(x, ...)
```

## Arguments

- x:

  fitted `"ptLasso"` object.

- ...:

  other arguments to pass to the print function.

## See also

`ptLasso`, `cv.ptLasso` and `predict.cv.ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group;

fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
print(fit)
#> 
#> Call:  ptLasso(x = x, y = y, groups = groups, family = "gaussian", type.measure = "mse",      use.case = "inputGroups", group.intercepts = TRUE) 
#> 
```
