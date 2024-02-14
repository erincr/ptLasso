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
#' @param y \code{y} matrix as in \code{ptLasso}.
#' @param groups A vector of length nobs indicating to which group each observation belongs. For data with k groups, groups should be coded as integers 1 through k. 
#' @param alphalist A vector of values of the pretraining hyperparameter alpha. Defaults to \code{seq(0, 1, length.out=11)}.
#' @param family Response type as in \code{ptLasso}.
#' @param type.measure Measure computed in \code{cv.glmnet}, as in \code{ptLasso}.
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfold} identifying what fold each observation is in. If supplied, \code{nfold} can be missing.
#' @param s For \code{fit.method = "glmnet"} only. The choice of lambda to be used by all models when estimating the CV performance for each choice of alpha. Defaults to "lambda.min". May be "lambda.1se", or a numeric value. (Use caution when supplying a numeric value: the same lambda will be used for all models.)
#' @param which For \code{fit.method = "sparsenet"} only. The choice of lambda and gamma to be used by all models when estimating the CV performance for each choice of alpha. Defaults to "parms.min". May also be "parms.1se".
#' @param alphahat.choice When choosing alphahat, we may prefer the best performance using all data (\code{alphahat.choice = "overall"}) or the best average performance across groups (\code{alphahat.choice = "mean"}). This is particularly useful when \code{type.measure} is "auc" or "C". These measures look at pairwise comparisons, and therefore are likely to be quite different when using the entire dataset (all pairwise comparisons) and individual groups (comparisons within groups only).
#' @param use.case The type of grouping observed in the data. Can be one of "inputGroups" or "targetGroups".
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit.
#' @param fitall An optional cv.glmnet (or cv.sparsenet) object specifying the overall model.
#' @param fitind An optional list of cv.glmnet (or cv.sparsenet) objects specifying the individual models.
#' @param \dots Additional arguments to be passed to the cv.glmnet function. Some notable choices are \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{cv.glmnet}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. Importantly, \code{"ptLasso"} does not support the arguments \code{"intercept"}, \code{"offset"}, \code{"fit"} and \code{"check.args"}.
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
#'
#' @seealso \code{\link{ptLasso}} and \code{\link{plot.cv.ptLasso}}.
#' @examples
#' #### Gaussian example
#' # Train data
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' # Test data
#' outtest = gaussian.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#'
#' # Model fitting
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' cvfit
#' # plot(cvfit) # to see CV performance as a function of alpha 
#' predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
#'
#' #### Binomial example
#' # Train data
#' out = binomial.example.data()
#' x = out$x; y=out$y; groups = out$group
#'
#' # Test data
#' outtest = binomial.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#'
#' # Model fitting
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
#' outtest = binomial.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "binomial", type.measure = "mse", parallel=TRUE)
#' }
#' 
#' @export
cv.ptLasso <- function(x, y, groups = NULL, alphalist=seq(0,1,length=11),
                       family = c("gaussian", "multinomial", "binomial","cox"),  use.case=c("inputGroups","targetGroups"),
                       type.measure = c("default", "mse", "mae", "auc","deviance","class", "C"),
                       nfolds = 10, foldid = NULL,
                       verbose=FALSE,
                       fitall=NULL, fitind=NULL, # Do we need to name these? They are in ptLasso.
                       s = "lambda.min", which = "parms.min",
                       alphahat.choice = "overall",
                       fit.method = "glmnet",
                       ...) { 
     
    use.case = match.arg(use.case, c("inputGroups","targetGroups"), several.ok=FALSE)
    family = match.arg(family)
    type.measure = match.arg(type.measure)
        
    this.call <- match.call()

    if(!(family %in% names(this.call))) this.call$family = family
    if(!(type.measure %in% names(this.call))) this.call$type.measure = type.measure
    if(!(use.case %in% names(this.call))) this.call$use.case = use.case

    # Warnings
    if((use.case == "inputGroups") & is.null(groups)) stop("For the input grouped setting, groups must be supplied.")
    if(!is.null(groups)) k = length(table(groups))
    if(is.null(groups))  k = length(table(y))
       
    if(use.case == "targetGroups" & !(family %in% c("binomial", "multinomial"))){
        stop("Only the multinomial and binomial families are available for target grouped data.")
    }

    if(!(type.measure %in% c("class", "deviance")) & family == "multinomial"){
        type.measure = "class"
        message("Only class and deviance are available as type.measure for multinomial models; class used instead.")
    }

    if(type.measure == "auc" & family != "binomial"){
        type.measure = "deviance"
        message("Only the binomial family can use type.measure = auce. Deviance used instead")
    }
    
    if(type.measure == "class" & !(family %in% c("binomial", "multinomial"))){
        type.measure = "deviance"
        message("Only multinomial and binomial families can use type.measure = class. Deviance used instead.")
    }


    if(fit.method == "sparsenet") { parms = if(which == "parms.min"){ function(m) m$which.min } else { function(m) m$which.1se } }

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

    fit=vector("list",length(alphalist))
    fitpre = list()
    
    class.sizes=table(groups)
    
    errcvm=NULL
    ii=0
    for(alpha in alphalist){
        ii=ii+1
        if(verbose) {
            cat("",fill=TRUE)
            cat(c("alpha=",alpha),fill=TRUE)
        }
        
        fit[[ii]]<- ptLasso(x,y,groups,alpha=alpha,family=family,type.measure=type.measure, use.case=use.case, foldid=foldid, nfolds=nfolds,
                            fitall = fitall, fitind = fitind, verbose = verbose, fit.method = fit.method, ...)

        if(is.null(fitall)) fitall = fit[[ii]]$fitall
        if(is.null(fitind)) fitind = fit[[ii]]$fitind
        fitpre[[ii]] = fit[[ii]]$fitpre

        if(use.case == "targetGroups"){
            # Get cross-validated predictions from all of the models, so we can compute a cross-validated performance measure
            phat = do.call(cbind, lapply(fit[[ii]]$fitpre, function(m) m$fit.preval[, m$lambda == get.lamhat(m, s)]))
            err  = sapply(fit[[ii]]$fitpre, function(m) m$cvm[m$lambda == get.lamhat(m, s)])
            err  = c(mult.perf(phat, y, type.measure), mean(err), err) 
        }

        if(use.case == "inputGroups"){
            err=NULL
            pre.preds = rep(NA, nrow(x))
            if(family == "multinomial") pre.preds = matrix(NA, nrow = nrow(x), ncol = length(table(y)))
            for(i in 1:length(fitpre[[ii]])){
                m = fitpre[[ii]][[i]]

                if(fit.method == "glmnet"){
                    lamhat = get.lamhat(m, s)
                    err = c(err, m$cvm[m$lambda == lamhat])
                } else if(fit.method == "sparsenet") {
                    lamgam = parms(m)
                    err = c(err, m$cvm[lamgam[1], lamgam[2]])
                }

                if(family == "multinomial"){
                    pre.preds[groups == i, ] = m$fit.preval[, , m$lambda == lamhat]
                } else {
                    if(fit.method == "sparsenet"){
                        pre.preds[groups == i]   = m$fit.preval[,  lamgam[1], lamgam[2]]
                    } else {
                        pre.preds[groups == i]   = m$fit.preval[,   m$lambda == lamhat]
                    }
                }
            }
        
            if(family == "multinomial") dim(pre.preds) = c(dim(pre.preds), 1)
            err = c(as.numeric(assess.glmnet(pre.preds, newy = y, family=family)[type.measure]), mean(err), weighted.mean(err, w = table(groups)/length(groups)), err)
        }
        
        errcvm = rbind(errcvm,err)
        
    }

    res=cbind(alphalist, errcvm)
    
    if(fit[[1]]$call$use.case=="inputGroups")  colnames(res) = c("alpha", "overall", "mean", "wtdMean", paste("group", as.character(1:length(class.sizes)),sep=""))
    if(fit[[1]]$call$use.case=="targetGroups") colnames(res) = c("alpha", "overall", "mean", paste("group", as.character(1:length(class.sizes)),sep=""))
    
    alphahat=if(alphahat.choice == "mean") { alphalist[which.f(res[, "mean"])] } else { alphalist[which.f(res[, "overall"])] }
    varying.alphahat = sapply(1:k, function(kk) alphalist[which.f(res[, paste0("group", kk)])])
    
    if(use.case == "targetGroups"){
        # Individual models
        phat = do.call(cbind, lapply(fit[[1]]$fitind, function(m) m$fit.preval[, m$lambda == get.lamhat(m, s)]))
        err.ind = sapply(fit[[1]]$fitind, function(m) f(m$cvm)) 
        err.ind = c(mult.perf(phat, y, type.measure), mean(err.ind), err.ind) # weighted.mean(err.ind, w = table(groups)/length(groups)),

        # Overall model
        err.all = c(f(fit[[1]]$fitall$cvm), rep(NA, length(err.ind) - 1))
        
        names(err.ind) = names(err.all) = colnames(res)[2:ncol(res)]
    } else if(use.case == "inputGroups"){
        # Individual models
        ind.preds = rep(NA, nrow(x))
        if(family == "multinomial") ind.preds = matrix(NA, nrow = nrow(x), ncol = length(table(y)))
        err.ind = NULL
        for(i in 1:k){
            m = fit[[1]]$fitind[[i]]
            if(fit.method == "glmnet"){
                lamhat = get.lamhat(m, s)
                err.ind = c(err.ind, m$cvm[m$lambda == lamhat])
            } else if(fit.method == "sparsenet") {
                lamgam = parms(m)
                err.ind = c(err.ind, m$cvm[lamgam[1], lamgam[2]])
            }

            if(family == "multinomial"){
                ind.preds[groups == i, ] = m$fit.preval[, , m$lambda == lamhat]
            } else {
                if(fit.method == "sparsenet"){
                    ind.preds[groups == i]   = m$fit.preval[,  lamgam[1], lamgam[2]]
                } else {
                    ind.preds[groups == i]   = m$fit.preval[,   m$lambda == lamhat]
                }
            }   
        }
        if(family == "multinomial") dim(ind.preds) = c(dim(ind.preds), 1)
        err.ind = c(as.numeric(assess.glmnet(ind.preds, newy = y, family=family)[type.measure]),
                    mean(err.ind),
                    weighted.mean(err.ind, w = table(groups)/length(groups)),
                    err.ind)

        # Overall model
        err.all = NULL
        m = fit[[1]]$fitall
        if(fit.method == "glmnet") lamhat = get.lamhat(m, s)
        if(fit.method == "sparsenet") lamgam = parms(m)
        for(i in 1:k){
            if(family == "multinomial"){
                overall.preds = m$fit.preval[groups == i, , m$lambda == lamhat]
                dim(overall.preds) = c(dim(overall.preds), 1)
                err.all = c(err.all,
                            as.numeric(assess.glmnet(overall.preds, newy = subset.y(y, groups==i, family), family=family)[type.measure])
                            )
            } else {
                if(fit.method == "glmnet") {
                    err.all = c(err.all,
                                as.numeric(assess.glmnet(m$fit.preval[groups == i, m$lambda == lamhat], newy = subset.y(y, groups==i, family), family=family)[type.measure])
                                )
                } else if(fit.method == "sparsenet"){
                    err.all = c(err.all,
                                as.numeric(assess.glmnet(m$fit.preval[groups == i, lamgam[1], lamgam[2]], newy = subset.y(y, groups==i, family), family=family)[type.measure])
                                )
                }
            }
        }
        err.all = c(mean(err.all), weighted.mean(err.all, w = table(groups)/length(groups)), err.all)
        if(fit.method == "glmnet")    err.all = c(m$cvm[m$lambda == lamhat], err.all)
        if(fit.method == "sparsenet") err.all = c(m$cvm[lamgam[1], lamgam[2]], err.all)
        names(err.ind) = names(err.all) = colnames(res)[2:ncol(res)]                
    }

    
    
    out=enlist(
               errpre = res, errind = err.ind, errall = err.all,
               alphahat,
               varying.alphahat, 
               alphalist,
               call=this.call,
               use.case = use.case,
               family,
               type.measure,
               fitind = fitind,
               fitall = fitall,
        fit)
 
    if(fit.method == "glmnet")    out$fitall.lambda = fit[[1]]$fitall.lambda
    if(fit.method == "sparsenet") out$fitall.which = fit[[1]]$fitall.which

    class(out)="cv.ptLasso"
    return(out)
}
