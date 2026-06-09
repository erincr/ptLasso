# Simulate input grouped data (gaussian outcome) for testing with ptLasso.

No required arguments; used primarily for documentation. Simply calls
`makedata` with a reasonable set of features.

## Usage

``` r
gaussian.example.data(
  k = 5,
  class.sizes = rep(200, k),
  n = sum(class.sizes),
  scommon = 10,
  sindiv = rep(10, k),
  p = 2 * (sum(sindiv) + scommon),
  beta.common = 3 * (1:k),
  beta.indiv = rep(3, k),
  intercepts = rep(0, k),
  sigma = 20
)
```

## Arguments

- k:

  Default: 5.

- class.sizes:

  Default: rep(200, k).

- n:

  Default: sum(class.sizes).

- scommon:

  Default: 10.

- sindiv:

  Default: rep(10, k).

- p:

  Default: 2\*(sum(sindiv) + scommon).

- beta.common:

  Default: 3\*(1:k).

- beta.indiv:

  Default: rep(3, k).

- intercepts:

  Default: rep(0, k).

- sigma:

  Default: 20.

## Value

A list for data with 5 groups and a gaussian outcome, n=1000 and p=120:

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
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group
```
