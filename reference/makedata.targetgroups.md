# Simulate target grouped data for testing with ptLasso.

Simulate target grouped data for testing with ptLasso.

## Usage

``` r
makedata.targetgroups(
  n,
  p,
  scommon,
  sindiv,
  class.sizes,
  shift.common,
  shift.indiv
)
```

## Arguments

- n:

  Total number of observations to simulate.

- p:

  Total number of features to simulate.

- scommon:

  Number of features shared by all groups.

- sindiv:

  Vector of length k. The i^th entry indicates the number of features
  specific to group i.

- class.sizes:

  Vector of length k. The i^th entry indicates the number of
  observations in group i.

- shift.common:

  A list of length k (one for each group). Each entry of this list
  should be a vector of length scommon, containing the shifts for the
  scommon features. The i^th entry of this list will be added to the
  first scommon columns of x for observations in group i.

- shift.indiv:

  The shifts for the individual features, in the same form as
  shift.common.

## Value

A list:

- x:

  Simulated features, size n x p.

- y:

  Outcomes y, length n.

## See also

`cv.ptLasso`, `ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r

k = 5
class.sizes = rep(50, k)
n = sum(class.sizes)
scommon = 3
sindiv = rep(3, k)
p = 3*(sum(sindiv) + scommon)
shift.common  = lapply(seq(-.1, .1, length.out = k), function(i) rep(i, scommon))
shift.indiv = lapply(1:k, function(i) -shift.common[[i]])

out = makedata.targetgroups(n=n, p=p, scommon=scommon,
                            sindiv=sindiv, class.sizes=class.sizes,
                            shift.common=shift.common, shift.indiv=shift.indiv)
x = out$x; y=out$y

```
