# Get the coefficients from a fitted cv.ptLasso model.

Get the coefficients from a fitted cv.ptLasso model.

## Usage

``` r
# S3 method for class 'cv.ptLasso'
coef(
  object,
  model = c("all", "individual", "overall", "pretrain"),
  alpha = NULL,
  ...
)
```

## Arguments

- object:

  fitted `"cv.ptLasso"` object.

- model:

  string indicating which coefficients to retrieve. Must be one of
  "all", "individual", "overall" or "pretrain".

- alpha:

  value between 0 and 1, indicating which alpha to use. If `NULL`,
  return coefficients from all models. Only impacts the results for
  model = "all" or model = "pretrain".

- ...:

  other arguments to be passed to the `"coef"` function. May be e.g.
  `s = "lambda.min"`.

## See also

`cv.ptLasso`, `ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
set.seed(1234)
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group;

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
# Get all model coefficients.
names(coef(cvfit))
#> [1] "individual" "pretrain"   "overall"   

coef(cvfit, model = "overall") # Overall model only
#> 62 x 1 sparse Matrix of class "dgCMatrix"
#>             lambda.1se
#> (Intercept) -2.0911194
#> groups2      6.9480612
#>              .        
#>              .        
#>              .        
#>              2.7965296
#>              3.3203621
#>              .        
#>              0.5383593
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
#>              .        
length(coef(cvfit, model = "individual")) # List of coefficients for each group model
#> [1] 2
length(coef(cvfit, model = "pretrain", alpha = .5)) # List of coefficients for each group model
#> [1] 2
```
