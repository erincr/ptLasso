% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/simulate.R
\name{binomial.example.data}
\alias{binomial.example.data}
\title{Simulate input grouped data (binomial outcome) for testing with ptLasso.}
\usage{
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
}
\arguments{
\item{k}{Default: 3.}

\item{class.sizes}{Default: rep(100, k).}

\item{n}{Default: sum(class.sizes).}

\item{scommon}{Default: 5.}

\item{sindiv}{Default: rep(5, k).}

\item{p}{Default: 2*(sum(sindiv) + scommon).}

\item{beta.common}{Default: list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3)).}

\item{beta.indiv}{Default: lapply(1:k, function(i)  0.9 * beta.common[[i]]).}

\item{intercepts}{Default: rep(0,k).}

\item{sigma}{Default: NULL.}
}
\value{
A list for data with 5 groups and a binomial outcome, n=300 and p=40:
\item{x}{Simulated features, size n x p.}
\item{y}{Outcomes y, length n.}
\item{groups}{Vector of length n, indicating which observations belong to which group.}
\item{snr}{Gaussian outcome only: signal to noise ratio.}
\item{mu}{Gaussian outcome only: the value of y before noise is added.}
}
\description{
No required arguments; used primarily for documentation. Simply calls \code{makedata} with a reasonable set of features.
}
\examples{
out = binomial.example.data()
x = out$x; y=out$y; groups = out$group


}
\seealso{
\code{cv.ptLasso}, \code{ptLasso}.
}
\author{
Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@stanford.edu>
}
\keyword{classification}
\keyword{models}
\keyword{regression}
