#' Simulate input grouped data for testing with ptLasso.
#' 
#'
#' @param n Total number of observations to simulate.
#' @param p Total number of features to simulate.
#' @param k Number of groups.
#' @param scommon Number of features shared by all groups.
#' @param sindiv Vector of length k. The i^th entry indicates the number of features specific to group i.
#' @param class.sizes Vector of length k. The i^th entry indicates the number of observations in group i.
#' @param beta.common The coefficients for the common features. This can be a vector of length k, in which case, the i^th entry is the coefficient for all scommon features for group i. This can alternatively be a list of length k (one for each group). Each entry of this list should be a vector of length scommon, containing the coefficients for the scommon features. 
#' @param beta.indiv The coefficients for the individual features, in the same form as beta.common.
#' @param intercepts A vector of length k, indicating the intercept for each group. Default is 0.
#' @param sigma Only used for the Gaussian outcome. Should be a number greater than or equal to 0, used to modify the amount of noise added. Default is 0.
#' @param outcome May be '"gaussian"', '"binomial"' or '"multinomial"'.
#' @param mult.classes Number of classes to simulate for the multinomial setting.
#'
#' @return A list:
#' \item{x}{Simulated features, size n x p.}
#' \item{y}{Outcomes y, length n.}
#' \item{groups}{Vector of length n, indicating which observations belong to which group.}
#' \item{snr}{Gaussian outcome only: signal to noise ratio.}
#' \item{mu}{Gaussian outcome only: the value of y before noise is added.}
#' 
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso}, \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#'
#' # Data with a binary outcome:
#' k = 3
#' class.sizes = rep(100, k)
#' n = sum(class.sizes)
#' scommon = 5
#' sindiv = rep(5, k) 
#' p = 2*(sum(sindiv) + scommon)
#' beta.common = lapply(1:k, function(i)  c(-.5, .5, .3, -.9, .1))
#' beta.indiv = lapply(1:k, function(i)  0.9 * beta.common[[i]])
#' 
#' out = makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
#'                beta.common=beta.common, beta.indiv=beta.indiv,
#'                class.sizes=class.sizes, outcome="binomial")
#' x = out$x; y=out$y; groups = out$group
#'
#'
#' @export
#' @importFrom stats rnorm rbinom runif var
#'
makedata=function(n, p, k, scommon, sindiv, class.sizes, beta.common, beta.indiv, intercepts = rep(0, k), sigma = 0,
                  outcome = c("gaussian", "binomial", "multinomial"), mult.classes = 3){
    
  outcome=match.arg(outcome)

    
  #generates simulated data
  x = matrix(rnorm(n*p),n,p)
  y = mu=rep(NA,n)

  if(outcome %in% c("binomial", "gaussian")){
      start.features = scommon + 1
      start.groups = 1
      
      for(kk in 1:k){
          end.features = start.features + sindiv[kk] - 1
          end.groups   = start.groups + class.sizes[kk] - 1

          beta0 = rep(0,p)
          beta0[1:scommon] = unlist(beta.common[kk])
          
          beta = rep(0,p)
          beta[start.features:end.features] = unlist(beta.indiv[kk])
          
          mu[start.groups:end.groups] = x[start.groups:end.groups, ] %*% (beta0+beta) + intercepts[kk]
          
          if(outcome == "gaussian"){
              y[start.groups:end.groups]  = mu[start.groups:end.groups] + sigma*rnorm(class.sizes[kk])
          } else {
              y[start.groups:end.groups]  = rbinom(class.sizes[kk], 1, prob = 1/(1 + exp(-mu[start.groups:end.groups])))
          }
          
          start.features = end.features + 1
          start.groups = end.groups + 1
      }
      
      snr=var(mu)/var(y-mu)
  } else {
      y = matrix(NA, nrow=n, ncol=mult.classes)
      start.features = scommon + 1
      for(cl in 1:mult.classes){
          start.groups = 1
          for(kk in 1:k){
              beta0 = rep(0,p)
              beta0[1:scommon] = beta.common[kk]
              
              beta = rep(0, p)
              end.features = start.features + sindiv[kk] - 1
              end.groups   = start.groups + class.sizes[kk] - 1

              beta[start.features:end.features] = beta.indiv[kk]
              
              y[start.groups:end.groups, cl] = x[start.groups:end.groups, ] %*% (beta0+beta) 
                            
              start.features = end.features + 1
              start.groups = end.groups + 1
          }
      }
      y = apply(y, 1, which.max)
      snr=var(mu)/var(y-mu)
  }
  
  
  groups = c(unlist(sapply(1:k, function(i) rep(i, class.sizes[i]))))

    if(outcome == "gaussian") return(enlist(x, y, snr, mu, groups))
    return(enlist(x, y, groups))
  
}



#' Simulate target grouped data for testing with ptLasso.
#' 
#'
#' @param n Total number of observations to simulate.
#' @param p Total number of features to simulate.
#' @param scommon Number of features shared by all groups.
#' @param sindiv Vector of length k. The i^th entry indicates the number of features specific to group i.
#' @param class.sizes Vector of length k. The i^th entry indicates the number of observations in group i.
#' @param shift.common  A list of length k (one for each group). Each entry of this list should be a vector of length scommon, containing the shifts for the scommon features. The i^th entry of this list will be added to the first scommon columns of x for observations in group i. 
#' @param shift.indiv The shifts for the individual features, in the same form as shift.common.
#'
#' @return A list:
#' \item{x}{Simulated features, size n x p.}
#' \item{y}{Outcomes y, length n.}
#' 
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso}, \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#' 
#' k = 5
#' class.sizes = rep(50, k)
#' n = sum(class.sizes)
#' scommon = 3
#' sindiv = rep(3, k)
#' p = 3*(sum(sindiv) + scommon)
#' shift.common  = lapply(seq(-.1, .1, length.out = k), function(i) rep(i, scommon))
#' shift.indiv = lapply(1:k, function(i) -shift.common[[i]])
#'
#' out = makedata.targetgroups(n=n, p=p, scommon=scommon,
#'                             sindiv=sindiv, class.sizes=class.sizes,
#'                             shift.common=shift.common, shift.indiv=shift.indiv)
#' x = out$x; y=out$y
#'
#'
#' @export
#' @importFrom stats rnorm 
makedata.targetgroups=function(n, p, scommon, sindiv, class.sizes, shift.common, shift.indiv){  
    x=matrix(rnorm(n*p),n,p)

    y=c(sapply(1:length(class.sizes), function(i) rep(i, class.sizes[i])))
   

    start.features = scommon + 1
    for(i in 1:length(class.sizes)){
        
        end.features = start.features + sindiv[i] - 1
        
        x[y == i, 1:scommon] = x[y == i,1:scommon] + unlist(shift.common[[i]])  #common features
        x[y == i, start.features:end.features] = x[y == i, start.features:end.features] + unlist(shift.indiv[[i]])  #  indiv features
        
        start.features = end.features + 1
    }

    return(list(x=x,y=y))
}


#' Simulate input grouped data (gaussian outcome) for testing with ptLasso.
#'
#' No required arguments; used primarily for documentation. Simply calls \code{makedata} with a reasonable set of features.
#'
#' @param k Default: 5.
#' @param class.sizes Default: rep(200, k).
#' @param n Default: sum(class.sizes).
#' @param scommon Default: 10.
#' @param sindiv Default: rep(10, k).
#' @param p Default: 2*(sum(sindiv) + scommon).
#' @param beta.common Default: 3*(1:k).
#' @param beta.indiv Default: rep(3, k).
#' @param intercepts Default: rep(0, k).
#' @param sigma Default: 20.
#' 
#' @return A list for data with 5 groups and a gaussian outcome, n=1000 and p=120:
#' \item{x}{Simulated features, size n x p.}
#' \item{y}{Outcomes y, length n.}
#' \item{groups}{Vector of length n, indicating which observations belong to which group.}
#' \item{snr}{Gaussian outcome only: signal to noise ratio.}
#' \item{mu}{Gaussian outcome only: the value of y before noise is added.}
#' 
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso}, \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group
#'
#' @export
#'
gaussian.example.data = function(k=5, class.sizes=rep(200, k), n=sum(class.sizes),
                                 scommon=10, sindiv=rep(10, k), p=2*(sum(sindiv) + scommon),
                                 beta.common= 3*(1:k), beta.indiv=rep(3, k), intercepts = rep(0, k), sigma=20){

    return(makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
                 class.sizes=class.sizes, beta.common=beta.common, beta.indiv=beta.indiv,
                 intercepts=intercepts, sigma=sigma, outcome="gaussian"))
}


#' Simulate input grouped data (binomial outcome) for testing with ptLasso.
#'
#' No required arguments; used primarily for documentation. Simply calls \code{makedata} with a reasonable set of features.
#' 
#' @param k Default: 3.
#' @param class.sizes Default: rep(100, k).
#' @param n Default: sum(class.sizes).
#' @param scommon Default: 5.
#' @param sindiv Default: rep(5, k).
#' @param p Default: 2*(sum(sindiv) + scommon).
#' @param beta.common Default: list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3)).
#' @param beta.indiv Default: lapply(1:k, function(i)  0.9 * beta.common[[i]]).
#' @param intercepts Default: rep(0,k).
#' @param sigma Default: NULL.
#' 
#' @return A list for data with 5 groups and a binomial outcome, n=300 and p=40:
#' \item{x}{Simulated features, size n x p.}
#' \item{y}{Outcomes y, length n.}
#' \item{groups}{Vector of length n, indicating which observations belong to which group.}
#' \item{snr}{Gaussian outcome only: signal to noise ratio.}
#' \item{mu}{Gaussian outcome only: the value of y before noise is added.}
#' 
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso}, \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#' out = binomial.example.data()
#' x = out$x; y=out$y; groups = out$group
#'
#'
#' @export
#'
binomial.example.data = function(k=3, class.sizes=rep(100, k), n = sum(class.sizes), scommon=5, sindiv=rep(5, k),
                                 p = 2*(sum(sindiv) + scommon),
                                 beta.common=list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3)),
                                 beta.indiv = lapply(1:k, function(i)  0.9 * beta.common[[i]]),
                                 intercepts=rep(0,k), sigma=NULL
                                 ){
    
    return(makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
                 class.sizes=class.sizes, beta.common=beta.common, beta.indiv=beta.indiv,
                 intercepts=intercepts, sigma=sigma, outcome="binomial"))
}
