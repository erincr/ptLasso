# Get the support for individual models

Get the indices of nonzero coefficients from the individual models in a
fitted ptLasso or cv.ptLasso object, excluding the intercept.

## Usage

``` r
get.individual.support(
  fit,
  s = "lambda.min",
  gamma = "gamma.min",
  commonOnly = FALSE,
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

- groups:

  which groups or responses to include when computing the support.
  Default is to include all groups/responses.

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
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group;

fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")

get.individual.support(fit) 
#>   [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
#>  [19]  19  20  21  22  23  25  27  28  29  30  31  32  34  35  36  37  38  39
#>  [37]  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57
#>  [55]  58  59  60  62  63  64  65  66  67  68  69  70  72  73  75  76  77  78
#>  [73]  79  80  82  83  84  87  88  89  90  91  92  93  94  95  98  99 100 101
#>  [91] 102 103 104 105 106 107 108 109 110 111 112 113 114 115 117 118 119 120

# only return features common to all groups 
get.individual.support(fit, commonOnly = TRUE) 
#>  [1]   1   2   3   4   5   6   7   8   9  10  11  13  14  37  38  48  51  54  60
#> [20]  65  67  73  80  82  83 108 118

# group 1 only
get.individual.support(fit, groups = 1) 
#>  [1]   1   2   3   4   5   6   7   8   9  10  11  13  14  15  16  18  20  22  29
#> [20]  31  32  34  37  38  47  48  50  54  57  58  59  60  62  64  66  67  70  72
#> [39]  73  77  79  80  83  88  89  91  93  94  95 101 103 104 106 108 111 113 114
#> [58] 115 117 118 119

cvfit = cv.ptLasso(x, y, groups = groups, alphalist = c(0, .5, 1), 
                   family = "gaussian", type.measure = "mse")

get.individual.support(cvfit)
#>   [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
#>  [19]  19  20  21  22  23  25  27  28  29  30  31  32  34  35  36  37  38  39
#>  [37]  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57
#>  [55]  58  59  60  61  62  63  64  65  66  67  68  69  70  72  73  75  76  77
#>  [73]  78  79  80  82  83  84  87  88  89  90  91  92  93  94  95  98  99 100
#>  [91] 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 117 118 119
#> [109] 120

# group 1 only
get.individual.support(cvfit, groups = 1) 
#>  [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  18  19  20
#> [20]  22  29  31  32  34  37  38  43  47  48  50  54  56  57  58  59  60  62  64
#> [39]  65  66  67  70  72  73  75  77  79  80  83  88  89  90  91  93  94  95 101
#> [58] 103 104 106 108 111 113 114 115 117 118 119
 
```
