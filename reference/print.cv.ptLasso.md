# Print the cv.ptLasso object.

Print the cv.ptLasso object.

## Usage

``` r
# S3 method for class 'cv.ptLasso'
print(x, ...)
```

## Arguments

- x:

  fitted `"cv.ptLasso"` object.

- ...:

  other arguments to pass to the print function.

## See also

`ptLasso`, `cv.ptLasso` and `predict.cv.ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group;

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
print(cvfit)
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
#>            alpha overall  mean wtdMean group_1 group_2
#> Overall            515.5 515.5   515.5   513.8   517.1
#> Pretrain     0.0   511.6 511.6   511.6   558.8   464.4
#> Pretrain     0.1   509.6 509.6   509.6   548.2   471.1
#> Pretrain     0.2   460.2 460.2   460.2   530.1   390.4
#> Pretrain     0.3   438.7 438.7   438.7   496.9   380.5
#> Pretrain     0.4   464.2 464.2   464.2   484.2   444.2
#> Pretrain     0.5   400.0 400.0   400.0   377.8   422.2
#> Pretrain     0.6   417.8 417.8   417.8   440.6   395.1
#> Pretrain     0.7   402.8 402.8   402.8   408.3   397.2
#> Pretrain     0.8   413.6 413.6   413.6   395.5   431.7
#> Pretrain     0.9   444.5 444.5   444.5   461.4   427.6
#> Pretrain     1.0   465.1 465.1   465.1   462.7   467.4
#> Individual         465.1 465.1   465.1   462.7   467.4
#> 
#> alphahat (fixed) = 0.5
#> alphahat (varying):
#> group_1 group_2 
#>     0.5     0.3 
```
