# Plot the cross-validation curve produced by cv.ptLasso, as a function of the `alpha` values used.

A plot is produced, and nothing is returned.

## Usage

``` r
# S3 method for class 'cv.ptLasso'
plot(x, plot.alphahat = FALSE, ...)
```

## Arguments

- x:

  Fitted `"cv.ptLasso"` object.

- plot.alphahat:

  If `TRUE`, show a dashed vertical line indicating the single value of
  alpha that maximized overall cross-validated performance.

- ...:

  Other graphical parameters to plot.

## See also

`ptLasso`, `cv.ptLasso` and `predict.cv.ptLasso`.

## Examples

``` r
set.seed(1234)
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
plot(cvfit) 
```
