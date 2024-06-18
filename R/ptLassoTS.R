#' Fit a pretrained lasso model using glmnet.
#'
#' Fits a pretrained lasso model using the glmnet package, for a fixed choice of the pretraining hyperparameter alpha. Additionally fits an "overall" model (using all data) and "individual" models (use each individual group). Can fit input-grouped data with Gaussian, multinomial, binomial or Cox outcomes, and target-grouped data, which necessarily has a multinomial outcome. Many ptLasso arguments are passed directly to glmnet, and therefore the glmnet documentation is another good reference for ptLasso.
#'
#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class '"sparseMatrix"' as in package 'Matrix'). Requirement: 'nvars >1'; in other words, 'x' should have 2 or more columns.
#' @param y response variable. Should me a matrix of dimension nobs x time points.
#' @param alpha The pretrained lasso hyperparameter, with \eqn{0\le\alpha\le 1}. The range of alpha is from 0 (which fits the overall model with fine tuning) to 1 (the individual models). The default value is 0.5, chosen mostly at random. To choose the appropriate value for your data, please either run \code{ptLasso} with a few choices of alpha and evaluate with a validation set, or use cv.ptLasso, which recommends a value of alpha using cross validation.
#' @param family Either a character string representing one of the built-in families, or else a 'glm()' family object. For more information, see Details section below or the documentation for response type.
#' @param type.measure loss to use for cross-validation within each individual, overall, or pretrained lasso model. Currently five options, not all available for all models. The default is 'type.measure="deviance"', which uses squared-error for gaussian models (a.k.a 'type.measure="mse"' there) and deviance for logistic regression. 'type.measure="class"' applies to binomial logistic regression only, and gives misclassification error. 'type.measure="auc"' is for two-class logistic regression only, and gives area under the ROC curve. 'type.measure="mse"' or 'type.measure="mae"' (mean absolute error) can be used by all; they measure the deviation from the fitted mean to the response.
#' @param overall.lambda The choice of lambda to be used by the overall model to define the offset and penalty factor for pretrained lasso. Defaults to "lambda.1se", could alternatively be "lambda.min". This choice of lambda will be used to compute the offset and penalty factor (1) during model training and (2) during prediction. In the predict function, another lambda must be specified for the individual models, the second stage of pretraining and the overall model.
#' @param overall.gamma For use only when the option \code{relax = TRUE} is specified. The choice of gamma to be used by the overall model to define the offset and penalty factor for pretrained lasso. Defaults to "gamma.1se", but "gamma.min" is also a good option. This choice of gamma will be used to compute the offset and penalty factor (1) during model training and (2) during prediction. In the predict function, another gamma must be specified for the individual models, the second stage of pretraining and the overall model.
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfolds} identifying what fold each observation is in. If supplied, \code{nfold} can be missing.
#' @param standardize Should the predictors be standardized before fitting (default is TRUE). 
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit with \code{cv.glmnet}.
#' @param weights observation weights. Default is 1 for each observation.
#' @param penalty.factor Separate penalty factors can be applied to each coefficient. This is a number that multiplies 'lambda' to allow differential shrinkage. Can be 0 for some variables,  which implies no shrinkage, and that variable is always included in the model. Default is 1 for all variables (and implicitly infinity for variables listed in 'exclude'). For more information, see \code{?glmnet}. For pretraining, the user-supplied penalty.factor will be multiplied by the penalty.factor computed by the overall model.
#' @param en.alpha The elasticnet mixing parameter, with 0 <= en.alpha <= 1. The penalty is defined as (1-alpha)/2||beta||_2^2+alpha||beta||_1. 'alpha=1' is the lasso penalty, and 'alpha=0' the ridge penalty. Default is `en.alpha = 1` (lasso).
#' @param \dots Additional arguments to be passed to the cv.glmnet functions. Notable choices include \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{"cv.glmnet"}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. ptLasso does not support the arguments \code{intercept}, \code{offset}, \code{fit} and \code{check.args}.
#'
#' 
#'
#' @return An object of class \code{"ptLasso"}, which is a list with the ingredients of the fitted models.
#' \item{call}{The call that produced this object.}
#' \item{nresps}{The number of responses.}
#' \item{alpha}{The value of alpha used for pretraining.}
#' \item{fitoverall}{A fitted \code{cv.glmnet} object trained using the full data.}
#' \item{fitpre}{A list of fitted (pretrained) \code{cv.glmnet} objects, one trained for each response.}
#' \item{fitind}{A list of fitted \code{cv.glmnet} objects, one trained with each group.}
#' \item{fitoverall.lambda}{Lambda used with fitoverall, to compute the offset for pretraining.}
#' 
#' @examples
#' \dontrun{
#' # Time series data with a continuous response
#' set.seed(1234)
#' n = 600; ntrain = 300; p = 50
#' x = matrix(rnorm(n*p), n, p)
#' sigma = 5
#'
#' beta1 = c(rep(2, 10), rep(0, p-10))
#' beta2 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 2), rep(0, p-15))
#' beta3 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 2), rep(1, 5), rep(0, p-20))
#'
#' y1 = x %*% beta1 + sigma * rnorm(n)
#' y2 = x %*% beta2 + sigma * rnorm(n)
#' y3 = x %*% beta2 + sigma * rnorm(n)
#' y = cbind(y1, y2, y3)
#'
#' fit = ptLasso(x, y, use.case = "timeSeries")
#' plot(fit)
#'
#' # Time series data with a binomial response
#' n = 600; ntrain = 300; p = 50
#' x = matrix(rnorm(n*p), n, p)
#' 
#' beta1 = c(rep(0.5, 10), rep(0, p-10))
#' beta2 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(0, p-15))
#' beta3 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(.5, 5), rep(0, p-20))
#' 
#' y1 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta1)))
#' y2 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta2)))
#' y3 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta3)))
#' y = cbind(y1, y2, y3)
#' 
#' fit =  ptLasso(x, y, verbose=T, use.case="timeSeries", family="binomial")
#' plot(fit)
#' 
#' }
#' 
#' @import glmnet Matrix
#' @seealso \code{\link{glmnet}}
#' @references Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1-22.
#'
#' @noRd
ptLassoTS=function(x,y,groups,alpha=0.5,family=c("gaussian", "binomial"),
                 type.measure=c("default", "mse", "mae", "auc","deviance","class"),
                 overall.lambda = c("lambda.1se", "lambda.min"),
                 overall.gamma = "gamma.1se",
                 foldid=NULL,
                 nfolds=10,
                 standardize = TRUE,
                 verbose=FALSE,
                 weights=NULL,
                 penalty.factor = rep(1, p),
                 en.alpha = 1,
                 fitind = NULL,
                 fitfirst=NULL,
                 call = NULL,
                 ...
                 ) {
    if(is.null(call)){ this.call = match.call() } else { this.call = call }
    
    family = match.arg(family)
    if(family == "multinomial") stop("ptLasso does not support multinomial responses for time series data.") # Sanity check this
    if(family == "cox") stop("ptLasso does not support survival responses for time series data.") # Sanity check this
        
    type.measure = match.arg(type.measure)
    if(type.measure == "default") type.measure = if(family == "gaussian") { "mse" } else { "deviance" }
    overall.lambda = match.arg(overall.lambda, c("lambda.1se", "lambda.min"))
    
    this.call$family = family
    this.call$use.case = "timeSeries"
    this.call$type.measure = type.measure    
   
    
    ############################################################################################################
    # Begin error checking:
    ############################################################################################################

    nresps = ncol(y)
    if(is.list(x)){
        isok = check.list.dims(x, nresps)
        if(!isok[[1]]) stop(isok[[2]])
        np = dim(x[[1]])
    } else {
        np=dim(x)
        if(is.null(np)|(np[2]<=1)) stop("x should be a matrix with 2 or more columns")
    }
    p = as.integer(np[2])
    n = as.integer(np[1])
    
    for(argument in c("fit", "check.args", "offset", "intercept", "standardize.response")){
        if(argument %in% names(list(...))) stop(paste0("ptLasso does not support the argument '", argument, "'."))
    }
    
    if((alpha > 1) | (alpha < 0)) stop("alpha must be between 0 and 1")
    
    # In the future, we want to be able to pass in just the predictions from the overall model.
    # This will be useful for settings where e.g. genentech has released a model (but maybe not as a glmnet object).
    if(!is.null(fitind)){
        if(length(fitind) != nresps) stop("Some of the individual models are missing: need one model trained for each response.")
        if(!(all(sapply(fitind, function(mm) "cv.glmnet" %in% class(mm))))) stop("fitind must be a list of cv.glmnet objects.")
        if(!all(sapply(fitind, function(mm) "fit.preval" %in% names(mm)))) stop("Individual models must have fit.preval defined (fitted with the argument keep = TRUE).")
         if(!all(sapply(fitind, function(mm) nrow(get.preval(mm, gamma = overall.gamma))) == n)) stop("Individual models must have been trained using the same training data passed to ptLasso.")
    }

    if(!is.null(fitfirst)){
        if(!("cv.glmnet" %in% class(fitfirst))) stop("fitfirst must be a cv.glmnet object.")
        if(!("fit.preval" %in% names(fitfirst))) stop("fitfirst must have fit.preval defined (fitted with the argument keep = TRUE).")
         if(!nrow(get.preval(fitfirst, gamma = overall.gamma)) == n) stop("First model must have been trained using the same training data passed to ptLasso.")
    }

    type.measure = cvtype(type.measure=type.measure,family=family)
    this.call$type.measure = type.measure
    
    ############################################################################################################
    # End error checking
    ############################################################################################################
    intercept=TRUE  
    
    if(is.null(foldid)){ 
        foldid = sample(rep(1:nfolds, ceiling(n/nfolds))[1:n])
    }

    fitind.is.null = is.null(fitind)
    
    ############################################################################################################
    # Fit models
    ############################################################################################################

    # Store models
    if(fitind.is.null) fitind = vector("list", nresps)
    fitpre = vector("list", nresps)

    # Initialize offset and penalty factor
    offset = NULL
    pf = penalty.factor
    supp = NULL
    for(i in 1:nresps){
        this.x = x
        if(is.list(x)) this.x = x[[i]]
        
        if(verbose) {
            txt = if(fitind.is.null){ "pretrained and individual models" } else { "pretrained model"}
            cat("\tFitting", txt, "for response", i, "/", nresps, fill=TRUE)
        }
        
        # The first model is the same as the individual model
        if( (i == 1) & (!fitind.is.null) ) fitpre[[1]] = fitind[[1]]
        if( (i == 1) & !is.null(fitfirst) ) { fitpre[[1]] = fitfirst; fitind[[1]] = fitfirst; }

        if(is.null(fitpre[[i]])){
            fitpre[[i]] = cv.glmnet(
                    this.x, y[, i],
                    offset=offset,
                    penalty.factor=pf,
                    family=family,
                    type.measure=type.measure,
                    foldid=foldid,
                    intercept=intercept,
                    weights=weights,
                    keep=TRUE,
                    standardize=standardize,
                    alpha=en.alpha,
                    ...
                )
        }
        if(fitind.is.null){
            if(i == 1) {
                fitind[[i]] = fitpre[[i]]
            } else {
                    fitind[[i]] = cv.glmnet(
                        this.x, y[, i],
                        family=family,
                        type.measure=type.measure,
                        foldid=foldid,
                        intercept=intercept,
                        penalty.factor=penalty.factor,
                        weights=weights,
                        keep=TRUE,
                        standardize=standardize,
                        alpha=en.alpha,
                        ...
                    )
            }
        }
        

        if(i < nresps){
            # Update penalty.factor and offset for the next round
            if(overall.lambda == "lambda.min") lamhat = fitpre[[i]]$lambda.min
            if(overall.lambda == "lambda.1se") lamhat = fitpre[[i]]$lambda.1se
            preval.offset = get.preval(fitpre[[i]], gamma = overall.gamma)[, fitpre[[i]]$lambda == lamhat]
            bhatall = as.numeric(coef(fitpre[[i]], s=lamhat, exact=FALSE))
            this.supp = if(family == "cox") { which(bhatall != 0) } else { which(bhatall[-1] != 0) }
            supp = sort(unique(c(supp, this.supp)))
            
            pf = rep(1/alpha, p)
            pf[supp] = 1
            pf = pf * penalty.factor

            if((alpha == 0) & (length(supp) == 0)) {
                almost.zero = 1e-9
                pf = rep(1/almost.zero, p)
                pf[supp] = 1
                pf = penalty.factor * pf
            }

            # I think the preval fits already contain the offset
            # which means that preval.offset contains the offset from all previous models
            offset = (1 - alpha) * preval.offset
       }
    }

    out=enlist(
        # Info about the initial call:
        call=this.call,
        nresps, alpha, 
               
        # Fitted models
        fitind, fitpre,
        fitoverall.lambda = overall.lambda
    )
    if(("relax" %in% names(list(...))) && list(...)$relax == TRUE) out$fitoverall.gamma = overall.gamma
    class(out)="ptLasso"
    return(out)

}


#########################################################################
# CV
#########################################################################
#' Cross-validation for ptLassoTS
#'
#' Cross-validation for \code{ptLassoTS}.
#' @examples
#' \dontrun{
#' # Not run - these examples are in the cv.ptLasso documentation.
#' set.seed(1234)
#' n = 600; ntrain = 300; p = 50
#'
#' x = matrix(rnorm(n*p), n, p)
#'
#' beta1 = c(rep(0.5, 10), rep(0, p-10))
#' beta2 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(0, p-15))
#' beta3 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(.5, 5), rep(0, p-20))
#'
#' y1 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta1)))
#' y2 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta2)))
#' y3 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta3)))
#' y = cbind(y1, y2, y3)
#' cvfit =  cv.ptLasso(x, y, verbose=T, use.case="timeSeries", family="binomial")
#' }
#' 
#' @noRd
cv.ptLassoTS <- function(x, y, alphalist=seq(0,1,length=11),
                         family = c("gaussian", "binomial"),
                         type.measure=c("default", "mse", "mae", "auc","deviance","class"),
                         nfolds = 10, foldid = NULL,
                         verbose=FALSE,
                         fitind=NULL,
                         fitfirst=NULL,
                         s = "lambda.min",
                         gamma = "gamma.min",
                         call = NULL,
                         ...) { 

    if(is.null(call)){ this.call = match.call() } else { this.call = call }
    
    type.measure = match.arg(type.measure, c("default", "mse", "mae", "auc","deviance","class"))
    if(type.measure == "default") type.measure = if(family == "gaussian") { "mse" } else { "deviance" }

    this.call$type.measure = type.measure
    this.call$use.case = "timeSeries"

    if(length(alphalist) < 2) stop("Need more than one alpha in alphalist.")
        
    nresps = ncol(y)
    np = if(is.list(x)){ dim(x[[1]]) } else { dim(x) }
    if(is.null(np)|(np[2]<=1)) stop("Check the dimensions of x.")
    p = as.integer(np[2])
    n = as.integer(np[1])

    f=function(x){min(x)}; which.f = function(x){which.min(x)}
    if(type.measure=="auc") {
        f=function(x){max(x)}; which.f = function(x){which.max(x)}
    }

    fit = vector("list",length(alphalist))
    fitpre = list()
    
    errcvm=NULL
    ii=0
    for(alpha in alphalist){
        ii=ii+1
        if(verbose) {
            cat("",fill=TRUE)
            cat(c("alpha=",alpha),fill=TRUE)
        }
        
        fit[[ii]]<- ptLassoTS(x,y,alpha=alpha,family=family, type.measure=type.measure, foldid=foldid, nfolds=nfolds,
                              fitind = fitind, fitfirst = fitfirst, verbose = verbose, ...)
        fit[[ii]]$call$type.measure = type.measure
        
        if(is.null(fitind) & (ii == 1))   fitind = fit[[ii]]$fitind
        if(is.null(fitfirst) & (ii == 1)) fitfirst = fit[[ii]]$fitind[[1]]
        fitpre[[ii]] = fit[[ii]]$fitpre

        err=NULL
        pre.preds = array(NA, c(n, nresps, 1))
        for(i in 1:nresps){
            m = fitpre[[ii]][[i]]
            lamhat = get.lamhat(m, s)
            err = c(err, get.cvm(m, gamma = gamma)[m$lambda == lamhat])
            pre.preds[, i, 1]   = get.preval(m, gamma = gamma)[,   m$lambda == lamhat]
        }
        err = c(mean(err), err)
        errcvm = rbind(errcvm,err)
    }
    res=cbind(alphalist, errcvm)
    colnames(res) = c("alpha", "mean", paste("response_", as.character(1:nresps), sep=""))
    
    alphahat = alphalist[which.f(res[, "mean"])]
    varying.alphahat = sapply(1:nresps, function(kk) alphalist[which.f(res[, paste0("response_", kk)])])
    
    # Individual models
    ind.preds = array(NA, c(n, nresps, 1))
    err.ind = NULL
    for(i in 1:nresps){
        m = fit[[1]]$fitind[[i]]
        lamhat = get.lamhat(m, s)
        err.ind = c(err.ind, get.cvm(m, gamma = gamma)[m$lambda == lamhat])
        ind.preds[, i, 1]   = get.preval(m, gamma = gamma)[,   m$lambda == lamhat]  
    }
    err.ind = c(mean(err.ind), err.ind)

    names(err.ind) = colnames(res)[2:ncol(res)]                 

    this.call$type.measure = type.measure
    this.call$group.intercepts = FALSE
    
    out=enlist(
               errpre = res, errind = err.ind,
               alphahat,
               varying.alphahat, 
               alphalist,
               call=this.call,
               type.measure,
               fitind = fitind,
               fitoverall.lambda = fit[[1]]$fitoverall.lambda,
               fit)
    if("fitoverall.gamma" %in% names(fit[[1]])) out$fitoverall.gamma = fit[[1]]$fitoverall.gamma
    class(out)="cv.ptLasso"
    return(out)
}

#' Helper for error checking
#' @noRd
check.list.dims <- function(x, nresps) {
    if(length(x) != nresps){
        return(list(FALSE, "The length of x should be the same as the number of columns of y."))
    }

    np = dim(x[[1]])
    if(is.null(np)|(np[2]<=1)){
        return(list(FALSE, "Each entry of x should be a matrix with 2 or more columns."))
    }

    all.nrows = sapply(x, nrow)
    if(length(unique(all.nrows)) > 1){
        return(list(FALSE, "Each entry of x should have the same number of rows."))
    }

    all.ncols = sapply(x, ncol)
    if(length(unique(all.ncols)) > 1){
        return(list(FALSE, "Each entry of x should have the same number of columns."))
    }

    return(list(TRUE, "ok"))
    
}
