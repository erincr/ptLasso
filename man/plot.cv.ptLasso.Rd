% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.cv.ptLasso.R
\name{plot.cv.ptLasso}
\alias{plot.cv.ptLasso}
\title{Plot the cross-validation curve produced by cv.ptLasso, as a function of the \code{alpha} values used.}
\usage{
\method{plot}{cv.ptLasso}(x, plot.alphahat = FALSE, ...)
}
\arguments{
\item{x}{Fitted \code{"cv.ptLasso"} object.}

\item{plot.alphahat}{If \code{TRUE}, show a dashed vertical line indicating the single value of alpha that maximized overall cross-validated performance.}

\item{\dots}{Other graphical parameters to plot.}
}
\description{
A plot is produced, and nothing is returned.
}
\examples{
set.seed(1234)
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group

cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
plot(cvfit) 

}
\seealso{
\code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
}
\author{
Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@stanford.edu>
}
\keyword{classification}
\keyword{models}
\keyword{regression}
