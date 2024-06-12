#' @noRd
get.lamhat <- function(model, s = "lambda.min"){
    if(s == "lambda.min") return(model$lambda.min)
    if(s == "lambda.1se") return(model$lambda.1se)
    if(is.numeric(s)) return(s)
}
#' @noRd
subset.y <- function(y, ix, family) {
    if(family == "cox") return(y[ix, ])
    return(y[ix])
}

#' Cross-validation for ptLasso
#'
#' Cross-validation for \code{ptLasso}.
#'
#' This function runs \code{ptLasso} once for each requested choice of alpha, and returns the cross validated performance.
#'
#' @param x \code{x} matrix as in \code{ptLasso}.
#' @param y \code{y} vector as in \code{ptLasso}.
#' @param groups A vector of length nobs indicating to which group each observation belongs. For data with k groups, groups should be coded as integers 1 through k. 
#' @param alphalist A vector of values of the pretraining hyperparameter alpha. Defaults to \code{seq(0, 1, length.out=11)}. This function will do pretraining for each choice of alpha in alphalist and return the CV performance for each alpha.
#' @param family Response type as in \code{ptLasso}.
#' @param type.measure Measure computed in \code{cv.glmnet}, as in \code{ptLasso}.
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfolds} identifying what fold each observation is in. If supplied, \code{nfolds} can be missing.
#' @param s The choice of lambda to be used by all models when estimating the CV performance for each choice of alpha. Defaults to "lambda.min". May be "lambda.1se", or a numeric value. (Use caution when supplying a numeric value: the same lambda will be used for all models.)
#' @param gamma For use only when \code{relax = TRUE}. The choice of gamma to be used by all models when estimating the CV performance for each choice of alpha. Defaults to "gamma.min". May also be "gamma.1se".
#' @param alphahat.choice When choosing alphahat, we may prefer the best performance using all data (\code{alphahat.choice = "overall"}) or the best average performance across groups (\code{alphahat.choice = "mean"}). This is particularly useful when \code{type.measure} is "auc" or "C", because the average performance across groups is different than the performance with the full dataset. The default is "overall".
#' @param use.case The type of grouping observed in the data. Can be one of "inputGroups" or "targetGroups".
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit.
#' @param fitoverall An optional cv.glmnet object specifying the overall model. This should have been trained on the full training data, with the argument keep = TRUE.
#' @param fitind An optional list of cv.glmnet objects specifying the individual models. These should have been trained on the training data, with the argumnet keep = TRUE.
#' @param group.intercepts For 'use.case = "inputGroups"' only. If `TRUE`, fit the overall model with a separate intercept for each group. If `FALSE`, ignore the grouping and fit one overall intercept. Default is `TRUE`.
#' @param \dots Additional arguments to be passed to the `cv.glmnet` function. Notable choices include \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{cv.glmnet}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. Importantly, \code{"cv.ptLasso"} does not support the arguments \code{"intercept"}, \code{"offset"}, \code{"fit"} and \code{"check.args"}.
#' 
#' @return An object of class \code{"cv.ptLasso"}, which is a list with the ingredients of the cross-validation fit.
#' \item{call}{The call that produced this object.}
#' \item{alphahat}{Value of \code{alpha} that optimizes CV performance on all data.}
#' \item{varying.alphahat}{Vector of values of \code{alpha}, the kth of which optimizes performance for group k.}
#' \item{alphalist}{Vector of all alphas that were compared.}
#' \item{errall}{CV performance for the overall model.}
#' \item{errpre}{CV performance for the pretrained models (one for each \code{alpha} tried).}
#' \item{errind}{CV performance for the individual model.}
#' \item{fit}{List of \code{ptLasso} objects, one for each \code{alpha} tried.}
#' \item{fitoverall}{The fitted overall model used for the first stage of pretraining.}
#' \item{fitoverall.lambda}{The value of \code{lambda} used for the first stage of pretraining.}
#' \item{fitind}{A list containing one individual model for each group.}
#' \item{use.case}{The use case: "inputGroups" or "targetGroups".}
#' \item{family}{The family used.}
#' \item{type.measure}{The type.measure used.}
#'
#' @seealso \code{\link{ptLasso}} and \code{\link{plot.cv.ptLasso}}.
#' @examples
#' # Getting started. First, we simulate data: we need covariates x, response y and group IDs.
#' set.seed(1234)
#' x = matrix(rnorm(1000*20), 1000, 20)
#' y = rnorm(1000)
#' groups = sort(rep(1:5, 200))
#' 
#' xtest = matrix(rnorm(1000*20), 1000, 20)
#' ytest = rnorm(1000)
#' groupstest = sort(rep(1:5, 200))
#' 
#' # Model fitting
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' cvfit
#' # plot(cvfit) # to see CV performance as a function of alpha 
#' predict(cvfit, xtest, groupstest, s="lambda.min") # to predict with held out data
#' predict(cvfit, xtest, groupstest, s="lambda.min", ytest=ytest) # to also measure performance
#'
#' # By default, we used s = "lambda.min" to compute CV performance.
#' # We could instead use s = "lambda.1se":
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse",
#'                    s = "lambda.1se")
#'
#' # We could also use the glmnet option relax = TRUE:
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse",
#'                    relax = TRUE)
#' # And, as we did with lambda, we may want to specify the choice of gamma to compute CV performance:
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse",
#'                    relax = TRUE, gamma = "gamma.1se")
#'
#' # Note that the first stage of pretraining uses "lambda.1se" and "gamma.1se" by default.
#' # This behavior can be modified by specifying overall.lambda and overall.gamma;
#' # see the documentation for ptLasso for more information.
#' 
#' # Now, we are ready to simulate slightly more realistic data.
#' # This continuous outcome example has k = 5 groups, where each group has 200 observations.
#' # There are scommon = 10 features shared across all groups, and
#' # sindiv = 10 features unique to each group.
#' # n = 1000 and p = 120 (60 informative features and 60 noise features).
#' # The coefficients of the common features differ across groups (beta.common).
#' # In group 1, these coefficients are rep(1, 10); in group 2 they are rep(2, 10), etc.
#' # Each group has 10 unique features, the coefficients of which are all 3 (beta.indiv).
#' # The intercept in all groups is 0.
#' # The variable sigma = 20 indicates that we add noise to y according to 20 * rnorm(n). 
#' set.seed(1234)
#' k=5
#' class.sizes=rep(200, k)
#' scommon=10; sindiv=rep(10, k)
#' n=sum(class.sizes); p=2*(sum(sindiv) + scommon)
#' beta.common=3*(1:k); beta.indiv=rep(3, k)
#' intercepts=rep(0, k)
#' sigma=20
#' out = gaussian.example.data(k=k, class.sizes=class.sizes,
#'                             scommon=scommon, sindiv=sindiv,
#'                             n=n, p=p,
#'                             beta.common=beta.common, beta.indiv=beta.indiv,
#'                             intercepts=intercepts, sigma=20)
#' x = out$x; y=out$y; groups = out$group
#'
#' outtest = gaussian.example.data(k=k, class.sizes=class.sizes,
#'                                 scommon=scommon, sindiv=sindiv,
#'                                 n=n, p=p,
#'                                 beta.common=beta.common, beta.indiv=beta.indiv,
#'                                 intercepts=intercepts, sigma=20)
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#'
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' cvfit
#' # plot(cvfit) # to see CV performance as a function of alpha 
#' predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
#'
#' # Now, we repeat with a binomial outcome.
#' # This example has k = 3 groups, where each group has 100 observations.
#' # There are scommon = 5 features shared across all groups, and
#' # sindiv = 5 features unique to each group.
#' # n = 300 and p = 40 (20 informative features and 20 noise features).
#' # The coefficients of the common features differ across groups (beta.common),
#' # as do the coefficients specific to each group (beta.indiv).
#' set.seed(1234)
#' k=3
#' class.sizes=rep(100, k)
#' scommon=5; sindiv=rep(5, k)
#' n=sum(class.sizes); p=2*(sum(sindiv) + scommon)
#' beta.common=list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3))
#' beta.indiv = lapply(1:k, function(i)  0.9 * beta.common[[i]])
#' 
#' out = binomial.example.data(k=k, class.sizes=class.sizes,
#'                             scommon=scommon, sindiv=sindiv,
#'                             n=n, p=p,
#'                             beta.common=beta.common, beta.indiv=beta.indiv)
#' x = out$x; y=out$y; groups = out$group
#'
#' outtest = binomial.example.data(k=k, class.sizes=class.sizes,
#'                                 scommon=scommon, sindiv=sindiv,
#'                                 n=n, p=p,
#'                                 beta.common=beta.common, beta.indiv=beta.indiv)
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#'
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "binomial",
#'                    type.measure = "auc", nfolds=3, verbose=TRUE, alphahat.choice="mean")
#' cvfit
#' # plot(cvfit) # to see CV performance as a function of alpha 
#' predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.1se")
#'
#' \dontrun{
#' ### Model fitting with parallel = TRUE
#' require(doMC)
#' registerDoMC(cores = 4)
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "binomial",
#'                    type.measure = "auc", parallel=TRUE)
#' }
#' 
#' @export
cv.ptLasso <- function(x, y, groups = NULL, alphalist=seq(0,1,length=11),
                       family = c("gaussian", "multinomial", "binomial","cox"),  use.case=c("inputGroups","targetGroups"),
                       type.measure = c("default", "mse", "mae", "auc","deviance","class", "C"),
                       nfolds = 10, foldid = NULL,
                       verbose=FALSE,
                       fitoverall=NULL, fitind=NULL, 
                       s = "lambda.min",
                       gamma = "gamma.min",
                       alphahat.choice = "overall",
                       group.intercepts = TRUE,
                       ...) { 
     
    use.case = match.arg(use.case, c("inputGroups","targetGroups"), several.ok=FALSE)
    family = match.arg(family)
    type.measure = match.arg(type.measure)
        
    this.call <- match.call()

    if(!(family %in% names(this.call))) this.call$family = family
    if(!(type.measure %in% names(this.call))) this.call$type.measure = type.measure
    if(!(use.case %in% names(this.call))) this.call$use.case = use.case

    original.groups = groups
    legend = group.legend(original.groups)
    groups = transform.groups(groups, legend = legend)
    
    # Warnings
    if((use.case == "inputGroups") & is.null(groups)) stop("For the input grouped setting, groups must be supplied.")
    if(!is.null(groups)) k = length(table(groups))
    if(is.null(groups))  k = length(table(y))

    if(length(alphalist) < 2) stop("Need more than one alpha in alphalist.")
    
    n <- nrow(x)
    p <- ncol(x)

    f=function(x){min(x)}; which.f = function(x){which.min(x)}
    if(type.measure=="auc" | type.measure=="C") {
        f=function(x){max(x)}
        which.f = function(x){which.max(x)}
    }
    
    if(use.case == "targetGroups"){ # The input groups case is handled directly by glmnet   
        mult.perf = function(predmat,y,type.measure){
            predmat.expanded=predmat
            dim(predmat.expanded) <- c(dim(predmat.expanded), 1)
            as.numeric(assess.glmnet(predmat.expanded, newy=y, family="multinomial")[type.measure])
        }
    }

    fit = vector("list",length(alphalist))
    fitpre = list()
    
    class.sizes = table(groups)
    
    errcvm=NULL
    ii=0
    for(alpha in alphalist){
        ii=ii+1
        if(verbose) {
            cat("",fill=TRUE)
            cat(c("alpha=",alpha),fill=TRUE)
        }
        
        fit[[ii]]<- ptLasso(x,y,groups,alpha=alpha,family=family,type.measure=type.measure, use.case=use.case, foldid=foldid, nfolds=nfolds,
                            fitoverall = fitoverall, fitind = fitind, verbose = verbose, group.intercepts = group.intercepts, ...)
        fit[[ii]]$group.legend = legend
        
        type.measure = fit[[ii]]$call$type.measure
        
        if(is.null(fitoverall)) fitoverall = fit[[ii]]$fitoverall 
        if(is.null(fitind)) fitind = fit[[ii]]$fitind
        fitpre[[ii]] = fit[[ii]]$fitpre

        if(use.case == "targetGroups"){
            # Get cross-validated predictions from all of the models, so we can compute a cross-validated performance measure
            phat = do.call(cbind, lapply(fit[[ii]]$fitpre, function(m) get.preval(m, gamma = gamma)[, m$lambda == get.lamhat(m, s)]))
            err  = sapply(fit[[ii]]$fitpre, function(m) get.cvm(m, gamma = gamma)[m$lambda == get.lamhat(m, s)])
            err  = c(mult.perf(phat, y, type.measure), mean(err), err) 
        }

        if(use.case == "inputGroups"){
            err=NULL
            pre.preds = rep(NA, nrow(x))
            if(family == "multinomial") pre.preds = matrix(NA, nrow = nrow(x), ncol = length(table(y)))
            for(i in 1:length(fitpre[[ii]])){
                m = fitpre[[ii]][[i]]
                
                lamhat = get.lamhat(m, s)
                err = c(err, get.cvm(m, gamma = gamma)[m$lambda == lamhat])
                if(family == "multinomial"){
                    pre.preds[groups == i, ] = get.preval(m, gamma = gamma)[, , m$lambda == lamhat]
                } else {
                    pre.preds[groups == i]   = get.preval(m, gamma = gamma)[,   m$lambda == lamhat]
                }
            }
            
            if(family == "multinomial") dim(pre.preds) = c(dim(pre.preds), 1)
            err = c(as.numeric(assess.glmnet(pre.preds, newy = y, family=family)[type.measure]), mean(err), weighted.mean(err, w = table(groups)/length(groups)), err)
        }
        
        errcvm = rbind(errcvm,err)
        
    }

    res=cbind(alphalist, errcvm)
    
    if(fit[[1]]$call$use.case=="inputGroups")  colnames(res) = c("alpha", "overall", "mean", "wtdMean", paste("group_", as.character(legend),sep=""))
    if(fit[[1]]$call$use.case=="targetGroups") colnames(res) = c("alpha", "overall", "mean", paste("group_", as.character(legend),sep=""))
    
    alphahat=if(alphahat.choice == "mean") { alphalist[which.f(res[, "mean"])] } else { alphalist[which.f(res[, "overall"])] }
    varying.alphahat = sapply(legend, function(kk) alphalist[which.f(res[, paste0("group_", kk)])])
    
    if(use.case == "targetGroups"){
        # Individual models
        phat = do.call(cbind, lapply(fit[[1]]$fitind, function(m) get.preval(m, gamma = gamma)[, m$lambda == get.lamhat(m, s)]))
        err.ind = sapply(fit[[1]]$fitind, function(m) f(get.cvm(m, gamma = gamma))) 
        err.ind = c(mult.perf(phat, y, type.measure), mean(err.ind), err.ind) # weighted.mean(err.ind, w = table(groups)/length(groups)),

        # Overall model
        err.overall = c(f(get.cvm(fit[[1]]$fitoverall, gamma = gamma)), rep(NA, length(err.ind) - 1))
        
        names(err.ind) = names(err.overall) = colnames(res)[2:ncol(res)]
    } else if(use.case == "inputGroups"){
        # Individual models
        ind.preds = rep(NA, nrow(x))
        if(family == "multinomial") ind.preds = matrix(NA, nrow = nrow(x), ncol = length(table(y)))
        err.ind = NULL
        for(i in 1:k){
            m = fit[[1]]$fitind[[i]]
            lamhat = get.lamhat(m, s)
            err.ind = c(err.ind, get.cvm(m, gamma = gamma)[m$lambda == lamhat])
            if(family == "multinomial"){
                ind.preds[groups == i, ] = get.preval(m, gamma = gamma)[, , m$lambda == lamhat]
            } else {
                ind.preds[groups == i]   = get.preval(m, gamma = gamma)[,   m$lambda == lamhat]
            }   
        }
        if(family == "multinomial") dim(ind.preds) = c(dim(ind.preds), 1)
        err.ind = c(as.numeric(assess.glmnet(ind.preds, newy = y, family=family)[type.measure]),
                    mean(err.ind),
                    weighted.mean(err.ind, w = table(groups)/length(groups)),
                    err.ind)

        # Overall model
        err.overall = NULL
        m = fit[[1]]$fitoverall
        lamhat = get.lamhat(m, s)
        for(i in 1:k){
            if(family == "multinomial"){
                overall.preds = get.preval(m, gamma = gamma)[groups == i, , m$lambda == lamhat]
                dim(overall.preds) = c(dim(overall.preds), 1)
                err.overall = c(err.overall,
                            as.numeric(assess.glmnet(overall.preds, newy = subset.y(y, groups==i, family), family=family)[type.measure])
                            )
            } else {
                err.overall = c(err.overall,
                                as.numeric(assess.glmnet(get.preval(m, gamma = gamma)[groups == i, m$lambda == lamhat], newy = subset.y(y, groups==i, family), family=family)[type.measure])
                                )
            }
        }
        err.overall = c(mean(err.overall), weighted.mean(err.overall, w = table(groups)/length(groups)), err.overall)
        err.overall = c(get.cvm(m, gamma = gamma)[m$lambda == lamhat], err.overall)
        names(err.ind) = names(err.overall) = colnames(res)[2:ncol(res)]                
    }  

    this.call$type.measure = type.measure
    this.call$family = family
    this.call$use.case = use.case
    this.call$group.intercepts = group.intercepts
    
    out=enlist(
               errpre = res, errind = err.ind, erroverall = err.overall,
               alphahat,
               varying.alphahat, 
               alphalist,
               call=this.call,
               use.case = use.case,
               family,
               type.measure,
               fitind = fitind,
               fitoverall = fitoverall,
               fitoverall.lambda = fit[[1]]$fitoverall.lambda,
               fit)
    if("fitoverall.gamma" %in% names(fit[[1]])) out$fitoverall.gamma = fit[[1]]$fitoverall.gamma
    class(out)="cv.ptLasso"
    return(out)
}
