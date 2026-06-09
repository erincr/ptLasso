# Plot the models trained by a ptLasso object

A plot is produced, and nothing is returned.

## Usage

``` r
# S3 method for class 'ptLasso'
plot(x, ...)
```

## Arguments

- x:

  Fitted `"ptLasso"` object.

- ...:

  Other parameters to pass to the plot function.

## See also

`ptLasso`, `cv.ptLasso` and `predict.cv.ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
set.seed(1234)
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group

fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
plot(fit) 

```
