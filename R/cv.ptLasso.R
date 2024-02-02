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
#' This function runs \code{ptLasso} once for each choice of alpha, and returns the cross-validated performance computed by cv.glmnet within each call to ptLasso.
#'
#' @param x \code{x} matrix as in \code{ptLasso}.
#' @param y \code{y} matrix as in \code{ptLasso}.
#' @param groups \code{groups} vector as in \code{ptLasso}
#' @param alpha A vector of values of the pretraining hyperparameter alpha. Defaults to \code{seq(0, 1, length.out=11)}.
#' @param family Response type as in \code{ptLasso}.
#' @param type.measure Measure computed in \code{cv.glmnet}, as in \code{ptLasso}.
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit.
#' @param ... Other arguments that can be passed to \code{ptLasso}.
#'
#' @return An object of class \code{"cv.ptLasso"}, which is a list with the ingredients of the cross-validation fit.
#' \item{fitall}{A fitted \code{cv.glmnet} object trained using the full data.}
#' \item{fitpre}{A list of fitted (pretrained) \code{cv.glmnet} objects, one trained with each group.}
#' \item{fitind}{A list of fitted \code{cv.glmnet} objects, one trained with each group.}
#' \item{alphahat}{Value of \code{alpha} that optimizes CV performance on all data.}
#' \item{varying.alphahat}{Vector of values of \code{alpha}, the kth of which optimizes performance for group k.}
#' \item{alphalist}{Vector of all alphas that were compared.}
#' \item{errall}{CV performance for the overall model.}
#' \item{errpre}{CV performance for the pretrained models (one for each \code{alpha} tried).}
#' \item{errind}{CV performance for the individual model.}
#' \item{useCase}{Whether the objective is "inputGroups" or "targetGroups".}
#' \item{type.measure}{Measure computed in \code{cv.glmnet}.}
#' \item{family}{Response type.}
#' \item{fit}{List of \code{ptLasso} objects, one for each \code{alpha} tried.}
#'
#' @seealso \code{\link{ptLasso}} and \code{\link{plot.cv.ptLasso}}.
#' @examples
#' 1+1
#'
#' @export
cv.ptLasso <- function(x, y, w = rep(1,length(y)), alphalist=seq(0,1,length=11),
                       groups = NULL, family = c("gaussian", "multinomial", "binomial","cox"),  use.case=c("inputGroups","targetGroups"),
                       type.measure=c("default", "mse", "mae", "auc","deviance","class", "C"),
                       nfolds = 10, foldid = NULL, keep = FALSE, verbose=FALSE, fitall=NULL, fitind=NULL,
                       trace.it = FALSE, weights = NULL,
                       overall.lambda = "lambda.1se",
                       s = "lambda.min",
                       parallel = FALSE,
                       ...) { #TODO: confirm that all ... args are OK
     
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

        fit[[ii]]<- ptLasso(x,y,groups,alpha=alpha,family=family,type.measure=type.measure, use.case=use.case, foldid=NULL, nfolds=nfolds,
                            fitall = fitall, fitind = fitind, verbose = verbose, trace.it = trace.it, weights = weights,
                            overall.lambda =  overall.lambda,
                            parallel = parallel, ...)

        if(is.null(fitall)) fitall = fit[[ii]]$fitall
        if(is.null(fitind)) fitind = fit[[ii]]$fitind
        fitpre[[ii]] = fit[[ii]]$fitpre

        if(use.case == "targetGroups"){
            # Get cross-validated predictions from all of the models, so we can compute a cross-validated performance measure
            phat = do.call(cbind, lapply(fit[[ii]]$fitpre, function(m) m$fit.preval[, m$lambda == get.lamhat(m, s)]))
            err  = sapply(fit[[ii]]$fitpre, function(m) m$cvm[m$lambda == get.lamhat(m, s)])
            err  = c(mult.perf(phat, y, type.measure), mean(err), err) # weighted.mean(err, w = table(groups)/length(groups)),
        }

        if(use.case == "inputGroups"){
            err=NULL
            pre.preds = rep(NA, nrow(x))
            if(family == "multinomial") pre.preds = matrix(NA, nrow = nrow(x), ncol = length(table(y)))
            for(i in 1:length(fitpre[[ii]])){
                m = fitpre[[ii]][[i]]
                lamhat = get.lamhat(m, s)
                err = c(err, m$cvm[m$lambda == lamhat])

                if(family == "multinomial"){
                    pre.preds[groups == i, ] = m$fit.preval[, , m$lambda == lamhat]
                } else {
                    pre.preds[groups == i]   = m$fit.preval[,   m$lambda == lamhat]
                }
            }
        
            if(family == "multinomial") dim(pre.preds) = c(dim(pre.preds), 1)
            err = c(as.numeric(assess.glmnet(pre.preds, newy = y, family=family)[type.measure]), mean(err), weighted.mean(err, w = table(groups)/length(groups)), err)
        }
        
        errcvm = rbind(errcvm,err)
        
    }

    res=cbind(alphalist, errcvm)
    
    if(fit[[1]]$use.case=="inputGroups") colnames(res)=c("alpha","overall","mean", "wtdMean", paste("group", as.character(1:length(class.sizes)),sep=""))
    if(fit[[1]]$use.case=="targetGroups") colnames(res)=c("alpha","overall","mean", paste("group", as.character(1:length(class.sizes)),sep=""))
    
    alphahat=alphalist[which.f(res[, "overall"])]
    varying.alphahat = sapply(1:k, function(kk) alphalist[which.f(res[, paste0("group", kk)])])
    
    if(use.case == "targetGroups"){
        # Individual models
        phat = do.call(cbind, lapply(fit[[1]]$fitind, function(m) m$fit.preval[, m$lambda == get.lamhat(m, s)]))
        err.ind = sapply(fit[[1]]$fitind, function(m) f(m$cvm)) 
        err.ind = c(mult.perf(phat, y, type.measure), mean(err.ind), err.ind) # weighted.mean(err.ind, w = table(groups)/length(groups)),

        # Overall model
        err.all = c(f(fit[[1]]$fitall$cvm), rep(NA, length(err.ind) - 1))
        
        names(err.ind) = names(err.all) = colnames(res)[2:ncol(res)]
    } else {
        # Individual models
        ind.preds = rep(NA, nrow(x))
        if(family == "multinomial") ind.preds = matrix(NA, nrow = nrow(x), ncol = length(table(y)))
        err.ind = NULL
        for(i in 1:k){
            m = fit[[1]]$fitind[[i]]
            lamhat = get.lamhat(m, s)
            err.ind = c(err.ind, m$cvm[m$lambda == lamhat])

            if(family == "multinomial"){
                ind.preds[groups == i, ] = m$fit.preval[, , m$lambda == lamhat]
            } else {
                ind.preds[groups == i] = m$fit.preval[, m$lambda == lamhat]
            }
            
        }
        if(family == "multinomial") dim(ind.preds) = c(dim(ind.preds), 1)
        err.ind = c(as.numeric(assess.glmnet(ind.preds, newy = y, family=family)[type.measure]), mean(err.ind), weighted.mean(err.ind, w = table(groups)/length(groups)), err.ind)

        # Overall model
        err.all = NULL
        m = fit[[1]]$fitall
        lamhat = get.lamhat(m, s)
        for(i in 1:k){
            if(family == "multinomial"){
                overall.preds = m$fit.preval[groups == i, , m$lambda == lamhat]
                dim(overall.preds) = c(dim(overall.preds), 1)
                err.all = c(err.all,
                            as.numeric(assess.glmnet(overall.preds, newy = subset.y(y, groups==i, family), family=family)[type.measure])
                            )
            } else {
                err.all = c(err.all,
                            as.numeric(assess.glmnet(m$fit.preval[groups == i, m$lambda == lamhat], newy = subset.y(y, groups==i, family), family=family)[type.measure])
                            )
            }
        }
        err.all = c(m$cvm[m$lambda == lamhat], mean(err.all), weighted.mean(err.all, w = table(groups)/length(groups)), err.all)
        names(err.ind) = names(err.all) = colnames(res)[2:ncol(res)]                
    }

    
    
    out=enlist(fitall, fitind, fitpre,
               errpre = res, errind = err.ind, errall = err.all,
               alphahat,
               varying.alphahat, 
               alphalist,
               call=this.call,
               use.case = use.case,
               family,
               type.measure,
               fit)
    class(out)="cv.ptLasso"
    return(out)
}
