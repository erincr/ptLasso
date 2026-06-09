# Simulate input grouped data (binomial outcome) for testing with ptLasso.

No required arguments; used primarily for documentation. Simply calls
`makedata` with a reasonable set of features.

## Usage

``` r
binomial.example.data(
  k = 3,
  class.sizes = rep(100, k),
  n = sum(class.sizes),
  scommon = 5,
  sindiv = rep(5, k),
  p = 2 * (sum(sindiv) + scommon),
  beta.common = list(c(-0.5, 0.5, 0.3, -0.9, 0.1), c(-0.3, 0.9, 0.1, -0.1, 0.2), c(0.1,
    0.2, -0.1, 0.2, 0.3)),
  beta.indiv = lapply(1:k, function(i) 0.9 * beta.common[[i]]),
  intercepts = rep(0, k),
  sigma = NULL
)
```

## Arguments

- k:

  Default: 3.

- class.sizes:

  Default: rep(100, k).

- n:

  Default: sum(class.sizes).

- scommon:

  Default: 5.

- sindiv:

  Default: rep(5, k).

- p:

  Default: 2\*(sum(sindiv) + scommon).

- beta.common:

  Default: list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1,
  .2, -.1, .2, .3)).

- beta.indiv:

  Default: lapply(1:k, function(i) 0.9 \* beta.common\[\[i\]\]).

- intercepts:

  Default: rep(0,k).

- sigma:

  Default: NULL.

## Value

A list for data with 5 groups and a binomial outcome, n=300 and p=40:

- x:

  Simulated features, size n x p.

- y:

  Outcomes y, length n.

- groups:

  Vector of length n, indicating which observations belong to which
  group.

- snr:

  Gaussian outcome only: signal to noise ratio.

- mu:

  Gaussian outcome only: the value of y before noise is added.

## See also

`cv.ptLasso`, `ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
out = binomial.example.data()
x = out$x; y=out$y; groups = out$group

```
