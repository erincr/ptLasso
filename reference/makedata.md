# Simulate input grouped data for testing with ptLasso.

Simulate input grouped data for testing with ptLasso.

## Usage

``` r
makedata(
  n,
  p,
  k,
  scommon,
  sindiv,
  class.sizes,
  beta.common,
  beta.indiv,
  intercepts = rep(0, k),
  sigma = 0,
  outcome = c("gaussian", "binomial", "multinomial"),
  mult.classes = 3
)
```

## Arguments

- n:

  Total number of observations to simulate.

- p:

  Total number of features to simulate.

- k:

  Number of groups.

- scommon:

  Number of features shared by all groups.

- sindiv:

  Vector of length k. The i^th entry indicates the number of features
  specific to group i.

- class.sizes:

  Vector of length k. The i^th entry indicates the number of
  observations in group i.

- beta.common:

  The coefficients for the common features. This can be a vector of
  length k, in which case, the i^th entry is the coefficient for all
  scommon features for group i. This can alternatively be a list of
  length k (one for each group). Each entry of this list should be a
  vector of length scommon, containing the coefficients for the scommon
  features.

- beta.indiv:

  The coefficients for the individual features, in the same form as
  beta.common.

- intercepts:

  A vector of length k, indicating the intercept for each group. Default
  is 0.

- sigma:

  Only used for the Gaussian outcome. Should be a number greater than or
  equal to 0, used to modify the amount of noise added. Default is 0.

- outcome:

  May be '"gaussian"', '"binomial"' or '"multinomial"'.

- mult.classes:

  Number of classes to simulate for the multinomial setting.

## Value

A list:

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

# Data with a binary outcome:
k = 3
class.sizes = rep(100, k)
n = sum(class.sizes)
scommon = 5
sindiv = rep(5, k) 
p = 2*(sum(sindiv) + scommon)
beta.common = lapply(1:k, function(i)  c(-.5, .5, .3, -.9, .1))
beta.indiv = lapply(1:k, function(i)  0.9 * beta.common[[i]])

out = makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
               beta.common=beta.common, beta.indiv=beta.indiv,
               class.sizes=class.sizes, outcome="binomial")
x = out$x; y=out$y; groups = out$group

```
