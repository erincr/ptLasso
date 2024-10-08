% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/coef.ptLasso.R
\name{coef.ptLasso}
\alias{coef.ptLasso}
\title{Get the coefficients from a fitted ptLasso model.}
\usage{
\method{coef}{ptLasso}(object, model = c("all", "individual", "overall", "pretrain"), ...)
}
\arguments{
\item{object}{fitted \code{"ptLasso"} object.}

\item{model}{string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".}

\item{\dots}{other arguments to be passed to the \code{"coef"} function. May be e.g. \code{s = "lambda.min"}.}
}
\value{
Model coefficients. If \code{model = "overall"}, this function returns the output of \code{coef}. If \code{model} is "individual" or "pretrain", this function returns a list containing the results of \code{coef} for each group-specific model. If \code{model = "all"}, this returns a list containing all (overall, individual and pretrain) coefficients.
}
\description{
Get the coefficients from a fitted ptLasso model.
}
\examples{
# Train data
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group;

fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
# Get all model coefficients.
names(coef(fit))

coef(fit, model = "overall") # Overall model only
length(coef(fit, model = "individual")) # List of coefficients for each group model
length(coef(fit, model = "pretrain")) # List of coefficients for each group model

}
\seealso{
\code{ptLasso}.
}
\author{
Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@stanford.edu>
}
\keyword{classification}
\keyword{models}
\keyword{regression}
