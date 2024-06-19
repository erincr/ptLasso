#' Fit a pretrained lasso model using glmnet.
#'
#' Fits a pretrained lasso model using the glmnet package, for a fixed choice of the pretraining hyperparameter alpha. Additionally fits an "overall" model (using all data) and "individual" models (use each individual group). Can fit input-grouped data with Gaussian, multinomial, binomial or Cox outcomes, and target-grouped data, which necessarily has a multinomial outcome. Many ptLasso arguments are passed directly to glmnet, and therefore the glmnet documentation is another good reference for ptLasso.
#'
#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class '"sparseMatrix"' as in package 'Matrix'). Requirement: 'nvars >1'; in other words, 'x' should have 2 or more columns.
#' @param y quantitative response variable, of dimension nobs x nresponses. 
#' @param alpha The pretrained lasso hyperparameter, with \eqn{0\le\alpha\le 1}. The range of alpha is from 0 (which fits the overall model with fine tuning) to 1 (the individual models). The default value is 0.5, chosen mostly at random. To choose the appropriate value for your data, please either run \code{ptLasso} with a few choices of alpha and evaluate with a validation set, or use cv.ptLasso, which recommends a value of alpha using cross validation.
#' @param type.measure loss to use for cross-validation within each individual, overall, or pretrained lasso model. Choices are 'type.measure="mse"' (mean squared error), 'type.measure="mae"' (mean absolute error) and 'type.measure="deviance"'.
#' @param overall.lambda The choice of lambda to be used by the overall model to define the offset and penalty factor for pretrained lasso. Defaults to "lambda.1se", could alternatively be "lambda.min". This choice of lambda will be used to compute the offset and penalty factor (1) during model training and (2) during prediction. In the predict function, another lambda must be specified for the individual models, the second stage of pretraining and the overall model.
#' @param overall.gamma For use only when the option \code{relax = TRUE} is specified. The choice of gamma to be used by the overall model to define the offset and penalty factor for pretrained lasso. Defaults to "gamma.1se", but "gamma.min" is also a good option. This choice of gamma will be used to compute the offset and penalty factor (1) during model training and (2) during prediction. In the predict function, another gamma must be specified for the individual models, the second stage of pretraining and the overall model.
#' @param fitoverall An optional cv.glmnet object specifying the overall model. This should have been trained on the full training data, with the argument keep = TRUE.
#' @param fitind An optional list of cv.glmnet objects specifying the individual models. 
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
#' \item{alpha}{The value of alpha used for pretraining.}
#' \item{fitoverall}{A fitted \code{cv.glmnet} object trained using the full data.}
#' \item{fitpre}{A list of fitted (pretrained) \code{cv.glmnet} objects, one trained for each response.}
#' \item{fitind}{A list of fitted \code{cv.glmnet} objects, one trained for each response.}
#' \item{fitoverall.lambda}{Lambda used with fitoverall, to compute the offset for pretraining.}
#' 
#' @examples
#' \dontrun{
#' # Getting started. First, we simulate data: we need covariates x and multiresponse y.
#' set.seed(1234)
#' n = 1000; ntrain = 500;
#' p = 500
#' sigma = 2
#'      
#' x = matrix(rnorm(n*p), n, p)
#' beta1 = c(rep(1, 5), rep(0.5, 5), rep(0, p - 10))
#' beta2 = c(rep(1, 5), rep(0, 5), rep(0.5, 5), rep(0, p - 15))
#' 
#' mu = cbind(x %*% beta1, x %*% beta2)
#' y  = cbind(mu[, 1] + sigma * rnorm(n), 
#'            mu[, 2] + sigma * rnorm(n))
#' cat("SNR for the two tasks:", round(diag(var(mu)/var(y-mu)), 2), fill=TRUE)
#' cat("Correlation between two tasks:", cor(y[, 1], y[, 2]), fill=TRUE)
#' 
#' xtest = x[-(1:ntrain), ]
#' ytest = y[-(1:ntrain), ]
#' 
#' x = x[1:ntrain, ]
#' y = y[1:ntrain, ]
#'
#' # Now, we can fit a ptLasso multiresponse model:
#' fit = ptLassoMult(x, y, alpha = 0.5, type.measure = "mse")
#' # plot(fit) # to see all of the cv.glmnet models trained
#' predict(fit, xtest) # to predict on new data
#' predict(fit, xtest, ytest=ytest) # if ytest is included, we also measure performance
#' }
#' 
#' @import glmnet Matrix
#' @seealso \code{\link{glmnet}}
#' @references Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1-22.
#'
#' @noRd
ptLassoMult=function(x,y,alpha=0.5,
                 type.measure=c("default", "mse", "mae", "deviance"),
                 overall.lambda = c("lambda.1se", "lambda.min"),
                 overall.gamma = "gamma.1se",
                 foldid=NULL,
                 nfolds=10,
                 standardize = TRUE,
                 verbose=FALSE,
                 weights=NULL,
                 penalty.factor = rep(1, nvars),
                 fitoverall=NULL, fitind=NULL,
                 en.alpha = 1,
                 call = NULL,
                 ...
                 ) {
    if(is.null(call)){ this.call = match.call() } else { this.call = call }
    
    this.call$use.case = "multiresponse"
    this.call$group.intercepts = FALSE
    
    type.measure = match.arg(type.measure, c("default", "mse", "mae", "deviance"))
    if(type.measure == "default") type.measure = "mse"
    this.call$type.measure = type.measure
    
    overall.lambda = match.arg(overall.lambda, c("lambda.1se", "lambda.min"))

    ############################################################################################################
    # Begin error checking:
    ############################################################################################################
    ##check dims
    np=dim(x) 
    if(is.null(np)|(np[2]<=1)) stop("x should be a matrix with 2 or more columns")
    nobs=as.integer(np[1])
    nvars=as.integer(np[2])

    npy=dim(y)
    if(is.null(npy)|(npy[2]<=1))stop("y should be a matrix with 2 or more columns")
    if(nobs != as.integer(npy[1])) stop("x and y should have the same number of observations")
    nresps=as.integer(npy[2])

    for(argument in c("fit", "check.args", "offset", "intercept", "standardize.response", "family")){
        if(argument %in% names(list(...))) stop(paste0("ptLasso does not support the argument '", argument, "'."))
    }
    
    if((alpha > 1) | (alpha < 0)) stop("alpha must be between 0 and 1")
    
    # In the future, we want to be able to pass in just the predictions from the overall model.
    # This will be useful for settings where e.g. genentech has released a model (but maybe not as a glmnet object).
    if(!is.null(fitoverall)){
        if(!("cv.glmnet" %in% class(fitoverall))) stop("fitoverall must be a cv.glmnet object.")
        if(!("fit.preval" %in% names(fitoverall))) stop("fitoverall must have fit.preval defined (fitted with the argument keep = TRUE).")
        if(nrow(get.preval(fitoverall, gamma = overall.gamma)) != nrow(x)) stop("fitoverall must have been trained using the same training data passed to ptLasso.")
    }
    if(!is.null(fitind)){
        if(length(fitind) != nresps) stop("Some of the individual models are missing: need one model trained for each response.")
        if(!(all(sapply(fitind, function(mm) "cv.glmnet" %in% class(mm))))) stop("fitind must be a list of cv.glmnet objects.")
        if(!all(sapply(fitind, function(mm) "fit.preval" %in% names(mm)))) stop("Individual models must have fit.preval defined (fitted with the argument keep = TRUE).")
        if(!all(sapply(fitind, function(mm) nrow(get.preval(mm, gamma = overall.gamma))) == nrow(x))) stop("Individual models must have been trained using the same training data passed to ptLasso.")
    }
    
    ############################################################################################################
    # End error checking
    ############################################################################################################
    p = ncol(x)
    
    if(is.null(foldid)){ 
        foldid = rep(sample(1:nfolds), ceiling(nobs/nfolds), replace = TRUE)
        foldid = foldid[1:nobs]
    }

    fitind.is.null = is.null(fitind)
    fitoverall.is.null = is.null(fitoverall)
    overall.pf = penalty.factor
    ############################################################################################################
    # Fit overall model 
    ############################################################################################################

    if(fitoverall.is.null){
        if(verbose) cat("Fitting overall model",fill=TRUE)

        fitoverall = cv.glmnet(x, y,
                            family="mgaussian",
                            foldid=foldid,
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            alpha=en.alpha,
                            ...)
    }

    if(overall.lambda == "lambda.min") lamhat = fitoverall$lambda.min
    if(overall.lambda == "lambda.1se") lamhat = fitoverall$lambda.1se
    preval.offset = get.preval(fitoverall, gamma = overall.gamma)[, , fitoverall$lambda == lamhat]
    bhatall = coef(fitoverall, s = lamhat)
    bhatall = do.call(cbind, bhatall)
    supall  = which((rowSums(bhatall) != 0)[-1])

    ############################################################################################################
    # Fit individual models
    # TODO: Combine these into one loop?
    ############################################################################################################

    if(verbose & fitind.is.null) cat("Fitting individual models",fill=TRUE)
    
    if(fitind.is.null){
        fitind=vector("list", nresps)
        
        for(kk in 1:nresps){
            
            # individual model 
            if(verbose) cat("\tFitting response", kk, "/", nresps, fill=TRUE)
            fitind[[kk]] = cv.glmnet(x, y[, kk],
                                        family="gaussian",
                                        type.measure=type.measure,
                                        foldid=foldid,
                                        penalty.factor=penalty.factor,
                                        weights=weights,
                                        keep=TRUE,
                                        standardize=standardize,
                                        alpha=en.alpha,
                                        ...)    
        }
    }
            
    ####################################################################################
    # Now, fit pretrained models 
    ####################################################################################
    if(verbose) cat("Fitting pretrained lasso models",fill=TRUE)
    
    fitpre=vector("list", nresps)

    if(alpha == 1){
        fitpre = fitind
    } else {
        for(kk in 1:nresps){
            if(verbose) cat("\tFitting pretrained model", kk, "/", nresps, fill=TRUE)

            offset = (1-alpha) * preval.offset[, kk]

            fac = rep(1/alpha, p)
            fac[supall] = 1
            pf = penalty.factor * fac 

            if((alpha == 0) & (length(supall) == 0)) {
                almost.zero = 1e-9
                fac = rep(1/almost.zero, p)
                pf = penalty.factor * fac
            }

            fitpre[[kk]] = cv.glmnet(x,
                                     y[, kk],
                                     family="gaussian",
                                     offset=offset,
                                     type.measure=type.measure,
                                     foldid=foldid,
                                     penalty.factor=pf,
                                     weights=weights,
                                     keep=TRUE,
                                     standardize=standardize,
                                     alpha=en.alpha,
                                     ...)
        }
    }

    out=enlist(
        # Info about the initial call:
        call=this.call,
        nresps, alpha, 
               
        # Fitted models
        fitoverall, fitind, fitpre,
        fitoverall.lambda = lamhat
    )
    if(("relax" %in% names(list(...))) && list(...)$relax == TRUE) out$fitoverall.gamma = overall.gamma
    class(out)="ptLasso"
    return(out)
}


#########################################################################
# CV
#########################################################################
#' Cross-validation for ptLassoMult
#'
#' Cross-validation for \code{ptLassoMult}.
#'
#' This function runs \code{ptLassoMult} once for each requested choice of alpha, and returns the cross validated performance.
#'
#' @param x \code{x} matrix as in \code{ptLassoMult}.
#' @param y \code{y} matrix as in \code{ptLassoMult}.
#' @param alphalist A vector of values of the pretraining hyperparameter alpha. Defaults to \code{seq(0, 1, length.out=11)}. This function will do pretraining for each choice of alpha in alphalist and return the CV performance for each alpha.
#' @param type.measure Measure computed in \code{cv.glmnet}, as in \code{ptLassoMult}.
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfolds} identifying what fold each observation is in. If supplied, \code{nfolds} can be missing.
#' @param s The choice of lambda to be used by all models when estimating the CV performance for each choice of alpha. Defaults to "lambda.min". May be "lambda.1se", or a numeric value. (Use caution when supplying a numeric value: the same lambda will be used for all models.)
#' @param gamma For use only when \code{relax = TRUE}. The choice of gamma to be used by all models when estimating the CV performance for each choice of alpha. Defaults to "gamma.min". May also be "gamma.1se".
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit.
#' @param fitoverall An optional cv.glmnet object specifying the multiresponse model. This should have been trained on the full training data, with the argument keep = TRUE.
#' @param fitind An optional list of cv.glmnet objects specifying the individual models. These should have been trained on the training data, with the argumnet keep = TRUE.
#' @param \dots Additional arguments to be passed to the `cv.glmnet` function. Notable choices include \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{cv.glmnet}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. Importantly, \code{"cv.ptLasso"} does not support the arguments \code{"intercept"}, \code{"offset"}, \code{"fit"} and \code{"check.args"}.
#' 
#' @return An object of class \code{"cv.ptLasso"}, which is a list with the ingredients of the cross-validation fit.
#' \item{call}{The call that produced this object.}
#' \item{alphahat}{Value of \code{alpha} that optimizes CV performance on all data.}
#' \item{varying.alphahat}{Vector of values of \code{alpha}, the kth of which optimizes performance for response k.}
#' \item{alphalist}{Vector of all alphas that were compared.}
#' \item{errall}{CV performance for the overall model.}
#' \item{errpre}{CV performance for the pretrained models (one for each \code{alpha} tried).}
#' \item{errind}{CV performance for the individual model.}
#' \item{fit}{List of \code{ptLassoMult} objects, one for each \code{alpha} tried.}
#' \item{fitoverall}{The fitted overall model used for the first stage of pretraining.}
#' \item{fitoverall.lambda}{The value of \code{lambda} used for the first stage of pretraining.}
#' \item{fitind}{A list containing one individual model for each group.}
#' \item{type.measure}{The type.measure used.}
#'
#' @seealso \code{\link{ptLassoMult}} and \code{\link{plot.cv.ptLasso}}.
#' @examples
#' \dontrun{
#' # Not run - these examples are in the cv.ptLasso documentation.
#' # Getting started. First, we simulate data: we need covariates x and multiresponse y.
#' set.seed(1234)
#' n = 1000; ntrain = 500;
#' p = 500
#' sigma = 2
#'      
#' x = matrix(rnorm(n*p), n, p)
#' beta1 = c(rep(1, 5), rep(0.5, 5), rep(0, p - 10))
#' beta2 = c(rep(1, 5), rep(0, 5), rep(0.5, 5), rep(0, p - 15))
#' 
#' mu = cbind(x %*% beta1, x %*% beta2)
#' y  = cbind(mu[, 1] + sigma * rnorm(n), 
#'            mu[, 2] + sigma * rnorm(n))
#' cat("SNR for the two tasks:", round(diag(var(mu)/var(y-mu)), 2), fill=TRUE)
#' cat("Correlation between two tasks:", cor(y[, 1], y[, 2]), fill=TRUE)
#' 
#' xtest = x[-(1:ntrain), ]
#' ytest = y[-(1:ntrain), ]
#' 
#' x = x[1:ntrain, ]
#' y = y[1:ntrain, ]
#'
#' # Now, we can fit a ptLasso multiresponse model:
#' fit = cv.ptLassoMult(x, y, type.measure = "mse")
#' # plot(fit) # to see the cv curve.
#' predict(fit, xtest) # to predict on new data
#' predict(fit, xtest, ytest=ytest) # if ytest is included, we also measure performance
#' # By default, we used s = "lambda.min" to compute CV performance.
#' # We could instead use s = "lambda.1se":
#' cvfit = cv.ptLassoMult(x, y, type.measure = "mse", s = "lambda.1se")
#'
#' # We could also use the glmnet option relax = TRUE:
#' cvfit = cv.ptLassoMult(x, y, type.measure = "mse", relax = TRUE)
#' # And, as we did with lambda, we may want to specify the choice of gamma to compute CV performance:
#' cvfit = cv.ptLassoMult(x, y, type.measure = "mse", relax = TRUE, gamma = "gamma.1se")
#'
#' # Note that the first stage of pretraining uses "lambda.1se" and "gamma.1se" by default.
#' # This behavior can be modified by specifying overall.lambda and overall.gamma;
#' # see the documentation for ptLasso for more information.
#' }
#' 
#' @noRd
cv.ptLassoMult <- function(x, y, alphalist=seq(0,1,length=11),
                       type.measure = c("default", "mse", "mae", "deviance"),
                       nfolds = 10, foldid = NULL,
                       verbose=FALSE,
                       fitoverall=NULL, fitind=NULL, 
                       s = "lambda.min",
                       gamma = "gamma.min",
                       call = NULL,
                       ...) { 
     
    type.measure = match.arg(type.measure, c("default", "mse", "mae", "deviance"))
    if(type.measure == "default") type.measure = "mse"
    
    this.call <- match.call()

    this.call$type.measure = type.measure
    this.call$use.case = "multiresponse"

    if(length(alphalist) < 2) stop("Need more than one alpha in alphalist.")
        
    nresps <- ncol(y)
    
    n <- nrow(x)
    p <- ncol(x)

    f=function(x){min(x)}; which.f = function(x){which.min(x)}

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
        
        fit[[ii]]<- ptLassoMult(x,y,alpha=alpha,type.measure=type.measure, foldid=foldid, nfolds=nfolds,
                                fitoverall = fitoverall, fitind = fitind, verbose = verbose, ...)
        fit[[ii]]$call$type.measure = type.measure
        
        if(is.null(fitoverall)) fitoverall = fit[[ii]]$fitoverall 
        if(is.null(fitind)) fitind = fit[[ii]]$fitind
        fitpre[[ii]] = fit[[ii]]$fitpre

        err=NULL
        pre.preds = array(NA, c(nrow(x), ncol(y), 1))
        for(i in 1:nresps){
            m = fitpre[[ii]][[i]]
            lamhat = get.lamhat(m, s)
            err = c(err, get.cvm(m, gamma = gamma)[m$lambda == lamhat])
            pre.preds[, i, 1]   = get.preval(m, gamma = gamma)[,   m$lambda == lamhat]
        }
        err = c(as.numeric(assess.glmnet(pre.preds, newy = y, family="mgaussian")[type.measure]), mean(err), err)
        errcvm = rbind(errcvm,err)
    }
    res=cbind(alphalist, errcvm)
    colnames(res) = c("alpha", "overall", "mean", paste("response_", as.character(1:nresps), sep=""))
    
    #alphahat=if(alphahat.choice == "mean") { alphalist[which.f(res[, "mean"])] } else { alphalist[which.f(res[, "overall"])] }
    alphahat = alphalist[which.f(res[, "mean"])]
    varying.alphahat = sapply(1:nresps, function(kk) alphalist[which.f(res[, paste0("response_", kk)])])
    
    # Individual models
    ind.preds = array(NA, c(nrow(x), ncol(y), 1))
    err.ind = NULL
    for(i in 1:nresps){
        m = fit[[1]]$fitind[[i]]
        lamhat = get.lamhat(m, s)
        err.ind = c(err.ind, get.cvm(m, gamma = gamma)[m$lambda == lamhat])
        ind.preds[, i, 1]   = get.preval(m, gamma = gamma)[,   m$lambda == lamhat]  
    }
    err.ind = c(as.numeric(assess.glmnet(ind.preds, newy = y, family="mgaussian")[type.measure]), mean(err.ind), err.ind)

    
    # Overall model
    err.overall = NULL
    m = fit[[1]]$fitoverall
    lamhat = get.lamhat(m, s)
    for(i in 1:nresps){
        err.overall = c(err.overall,
                        as.numeric(assess.glmnet(get.preval(m, gamma = gamma)[, i, m$lambda == lamhat], newy = y[, i], family="gaussian")[type.measure]))
    }
    err.overall = c(mean(err.overall), err.overall)
    err.overall = c(get.cvm(m, gamma = gamma)[m$lambda == lamhat], err.overall)
    names(err.ind) = names(err.overall) = colnames(res)[2:ncol(res)]                 

    this.call$type.measure = type.measure
    this.call$group.intercepts = FALSE
    
    out=enlist(
               errpre = res, errind = err.ind, erroverall = err.overall,
               alphahat,
               varying.alphahat, 
               alphalist,
               call=this.call,
               type.measure,
               fitind = fitind,
               fitoverall = fitoverall,
               fitoverall.lambda = fit[[1]]$fitoverall.lambda,
               fit)
    if("fitoverall.gamma" %in% names(fit[[1]])) out$fitoverall.gamma = fit[[1]]$fitoverall.gamma
    class(out)="cv.ptLasso"
    return(out)
}
