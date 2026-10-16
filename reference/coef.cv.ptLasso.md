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

## Value

Model coefficients. If `model = "overall"`, this returns the output of
`coef` for the overall model. If `model` is "individual", this returns a
list containing the results of `coef` for each group-specific (or
response-specific) model. If `model = "pretrain"`, this returns the
pretrained model coefficients: a list with one entry per value of
`alpha` if `alpha = NULL`, otherwise the coefficients for the chosen
`alpha`. If `model = "all"`, this returns a list containing all
(overall, individual and pretrain) coefficients. (For time series data,
there is no overall model, so only individual and pretrain coefficients
are returned.) Note that the pretrained coefficients are from the second
stage of pretraining only: they do not include the offset from the
overall model. The full pretrained linear predictor for a group (or
response) is `(1 - alpha)` times the overall model's linear predictor
(at `fitoverall.lambda`), plus the linear predictor from these
coefficients. For time series data, the offset instead comes from the
pretrained model for the previous time point.

## See also

`cv.ptLasso`, `ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<ercr@umich.edu\>

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
