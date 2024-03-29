
#' Predict using a cv.ptLasso object. 
#'
#' Return predictions and performance measures for a test set. 
#' 
#' @aliases predict.cv.ptLasso
#' @param cvfit Fitted \code{"cv.ptLasso"} object.
#' @param xtest Input matrix, matching the form used by \code{"cv.ptLasso"} for model training.
#' @param groupstest  A vector indicating to which group each observation belongs. Coding should match that used for model training. Will be NULL for target grouped data.
#' @param ytest Response variable. Optional. If included, \code{"predict"} will compute performance measures for xtest using code{"type.measure"} from the cvfit object.
#' @param alpha The chosen alpha to use for prediction. May be a vector containing one value of alpha for each group. If NULL, this will rely on the choice of "alphatype".
#' @param alphatype Choice of '"fixed"' or '"varying"'. If '"fixed"', use the alpha that achieved best cross-validated performance. If '"varying"', each group uses the alpha that optimized the group-specific cross-validated performance.
#' @param type Type of prediction required. Type '"link"' gives the linear predictors for '"binomial", '"multinomial"' or '"cox"' models; for '"gaussian"' models it gives the fitted values. Type '"response"' gives the fitted probabilities for '"binomial"' or '"multinomial"', and the fitted relative-risk for '"cox"'; for '"gaussian"' type '"response"' is equivalent to type '"link"'. Note that for '"binomial"' models, results are returned only for the class corresponding to the second level of the factor response. Type '"class"' applies only to '"binomial"' or '"multinomial"' models, and produces the class label corresponding to the maximum probability.
#' @param s Value of the penalty parameter 'lambda' at which predictions are required. Will use the same lambda for all models; can be a numeric value, '"lambda.min"' or '"lambda.1se"'. Default is '"lambda.min"'.
#' @param gamma For use only when 'relax = TRUE' was specified during training. Value of the penalty parameter 'gamma' at which predictions are required. Will use the same gamma for all models; can be a numeric value, '"gamma.min"' or '"gamma.1se"'. Default is '"gamma.min"'.
#' @param return.link If \code{TRUE}, will additionally return the linear link for the overall, pretrained and individual models: \code{linkoverall}, \code{linkpre} and \code{linkind}.
#' @return A list containing the requested predictions. If \code{ytest} is included, will also return error measures.
#' \item{call}{The call that produced this object.}
#' \item{alpha}{The value(s) of alpha used to generate predictions.}
#' \item{yhatoverall}{Predictions from the overall model.}
#' \item{yhatind}{Predictions from the individual models.}
#' \item{yhatpre}{Predictions from the pretrained models.}
#' \item{supoverall}{Indices of the features selected by the overall model.}
#' \item{supind}{Union of the indices of the features selected by the individual models.}
#' \item{suppre.common}{Features selected in the first stage of pretraining.}
#' \item{suppre.individual}{Union of the indices of the features selected by the pretrained models, without the features selected in the first stage.}
#' \item{type.measure}{If \code{ytest} is supplied, the performance measure computed.}
#' \item{erroverall}{If \code{ytest} is supplied, performance for the overall model. This is a named vector containing performance for (1) the entire dataset, (2) the average performance across groups, (3) the average performance across groups weighted by group size and (4) group-specific performance.}
#' \item{errind}{If \code{ytest} is supplied, performance for the overall model. As described in \code{erroverall}.}
#' \item{errpre}{If \code{ytest} is supplied, performance for the overall model. As described in \code{erroverall}.}
#' \item{linkoverall}{If \code{return.link} is TRUE, return the linear link from the overall model.}
#' \item{linkind}{If \code{return.link} is TRUE, return the linear link from the individual models.}
#' \item{linkpre}{If \code{return.link} is TRUE, return the linear link from the pretrained models.}
#'
#'
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' #### Gaussian example
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#' outtest = gaussian.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups;
#'
#' # Model fitting
#' # By default, use the single value of alpha that had the best CV performance on the entire dataset:
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
#' pred
#'
#' # For each group, use the value of alpha that had the best CV performance for that group:
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min", alphatype = "varying")
#' pred
#'
#' # Specify a single value of alpha and use lambda.1se.
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.1se",
#'                alphatype = "varying", alpha = .3)
#' pred
#'
#' # Specify a vector of choices for alpha: 
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min",
#'                alphatype = "varying", alpha = c(.1, .2, .3, .4, .5))
#' pred
#'
#' @import glmnet
#' @method predict cv.ptLasso
#' @export
predict.cv.ptLasso=function(cvfit, xtest,  groupstest=NULL, ytest=NULL, alpha=NULL, alphatype = c("fixed", "varying"),
                            type = c("link", "response", "class"), s = "lambda.min", gamma = "gamma.min", return.link = FALSE){

    if(missing("xtest")) stop("Please supply xtest.")

    type = match.arg(type, several.ok = FALSE)
    
    this.call = match.call()

    original.groups = groupstest
    legend = cvfit$fit[[1]]$group.legend
    groupstest = transform.groups(groupstest, legend = legend)
    
    alphatype = match.arg(alphatype)
    if(is.null(alpha)) {
        if(alphatype == "fixed")   alpha = cvfit$alphahat
        if(alphatype == "varying") alpha = cvfit$varying.alphahat
    }    

    close.enough = 1e-6
    if(length(alpha) == 1){      
        if(all(abs(alpha - cvfit$errpre[, "alpha"]) > close.enough)) stop("Not a valid choice of alpha. Please choose alpha from cvfit$alphalist.")
        
        model.ix = which(abs(cvfit$errpre[, "alpha"] - alpha) < close.enough)    
        if(length(model.ix) > 1) model.ix = model.ix[1]
       
        fit = cvfit$fit[[model.ix]]

        out = predict.ptLasso(fit, xtest, groupstest=original.groups, ytest=ytest, type = type, s = s, gamma = gamma)
        out$call = this.call
        
        return(out)
    } else {
        if(!all(sapply(alpha, function(x) any(abs(x - cvfit$errpre[, "alpha"]) < close.enough)))) {
            stop("Includes at least one invalid choice of alpha. Please choose alpha from cvfit$alphalist.")
        }
        
        # TODO future: what if they have only e.g. one group at prediction time?
        # They shouldn't have to supply alphas for every group.
        if(length(alpha) != cvfit$fit[[1]]$k) stop("Must have one alpha for each group.") 
        
        model.ix = sapply(alpha, function(a) which(abs(a - cvfit$errpre[, "alpha"]) < close.enough))
        model.ix = unname(model.ix)

        if(cvfit$call$use.case == "targetGroups") {
            results = lapply(1:cvfit$fit[[1]]$k,
                             function(ix) {predict.ptLasso.targetGroups(cvfit$fit[[model.ix[ix]]],
                                                           xtest, ytest = ytest,
                                                           pred.groups = ix,
                                                           process.results = FALSE,
                                                           type = type, s = s, gamma = gamma, return.link=TRUE)}
                             )
            phatall = results[[1]]$phatall
            phatpre = do.call(cbind, lapply(results, function(x) x$phatpre))
            phatind = do.call(cbind, lapply(results, function(x) x$phatind))
            suppre.common = results[[1]]$suppre.common
            suppre.individual = unique(sort(unlist(lapply(results, function(x) x$suppre.individual))))
            
            return(process.targetGroup.results(phatall = phatall, phatpre = phatpre, phatind = phatind,
                                        ytest = ytest, k = cvfit$fit[[1]]$k, type.measure = cvfit$type.measure,
                                        return.link = return.link, type = type,
                                        alpha = alpha, s = s, gamma = gamma, call = this.call, fit = cvfit,
                                        suppre.common = suppre.common, suppre.individual = suppre.individual
                                        )

                   )
        }

        
        predgroups = sort(unique(groupstest))

        results = lapply(predgroups,
                             function(ix) {predict.ptLasso(cvfit$fit[[model.ix[ix]]],
                                                           xtest[groupstest == ix, ],
                                                           groupstest=original.groups[groupstest == ix],
                                                           ytest=ytest[groupstest == ix],
                                                           type = type, s = s, gamma = gamma, return.link=TRUE)}
                             )
        
        all.preds = pre.preds = ind.preds = rep(NA, nrow(xtest))
        all.link = pre.link = ind.link = rep(NA, nrow(xtest))
        for(kk in 1:length(predgroups)){
            all.preds[groupstest == predgroups[kk]] = results[[kk]]$yhatoverall
            pre.preds[groupstest == predgroups[kk]] = results[[kk]]$yhatpre
            ind.preds[groupstest == predgroups[kk]] = results[[kk]]$yhatind

            all.link[groupstest == predgroups[kk]] = results[[kk]]$linkoverall
            pre.link[groupstest == predgroups[kk]] = results[[kk]]$linkpre
            ind.link[groupstest == predgroups[kk]] = results[[kk]]$linkind
        }
        
        suppre = lapply(predgroups, function(ix) get.pretrain.support(cvfit$fit[[model.ix[ix]]], groups = ix, includeOverall = FALSE, s = s, gamma = gamma))
        suppre = sort(unique(unlist(suppre)))
        if("fitoverall.gamma" %in% names(cvfit)){
            suppre.common =  get.overall.support(cvfit, s=cvfit$fitoverall.lambda, gamma = cvfit$fitoverall.gamma)
        } else {
            suppre.common =  get.overall.support(cvfit, s=cvfit$fitoverall.lambda)
        }
        suppre.individual = setdiff(suppre, suppre.common)
        
        if(!is.null(ytest)){
            group.weights = table(groupstest)/length(groupstest)
            erroverall = sapply(results, function(r) r$erroverall[grepl("group", names(r$erroverall))])
            erroverall = c(as.numeric(assess.glmnet(all.link, newy=ytest, family=cvfit$family)[cvfit$type.measure]),
                       mean(erroverall),
                       weighted.mean(erroverall, w = group.weights),
                       erroverall)

            errpre = sapply(results, function(r) r$errpre[grepl("group", names(r$errpre))])
            errpre = c(as.numeric(assess.glmnet(pre.link, newy=ytest, family=cvfit$family)[cvfit$type.measure]),
                       mean(errpre),
                       weighted.mean(errpre, w = group.weights),
                       errpre)

            errind = sapply(results, function(r) r$errind[grepl("group", names(r$errind))])
            errind = c(as.numeric(assess.glmnet(ind.link, newy=ytest, family=cvfit$family)[cvfit$type.measure]),
                       mean(errind),
                       weighted.mean(errind, w = group.weights),
                       errind)

            names(erroverall) = names(errpre) = names(errind) = c("overall", "mean", "wtdMean", paste0("group_", legend[predgroups]))
         }
        
        out = enlist(            
            yhatoverall = all.preds,
            yhatind = ind.preds, 
            yhatpre = pre.preds,

            suppre.common,
            suppre.individual,
            supoverall = results[[1]]$supoverall,
            supind = results[[1]]$supind,

            use.case = cvfit$fit[[1]]$call$use.case,

            type.measure = cvfit$fit[[1]]$call$type.measure,

            alpha,
            call = this.call
        )

        if(return.link){
            out$linkoverall = all.link
            out$linkind = ind.link
            out$linkpre = pre.link
        }
        
        if(!is.null(ytest)){
            out$errpre = errpre
            out$errind = errind
            out$erroverall = erroverall
        }
        
        class(out) = "predict.cv.ptLasso"
        
        return(out)
    }

    

}


#' Predict using a ptLasso object. 
#'
#' Return predictions and performance measures for a test set. 
#' 
#' @aliases predict.ptLasso
#' @param fit Fitted \code{"ptLasso"} object.
#' @param xtest Input matrix, matching the form used by \code{"ptLasso"} for model training.
#' @param groupstest  A vector indicating to which group each observation belongs. Coding should match that used for model training. Will be NULL for target grouped data.
#' @param ytest Response variable. Optional. If included, \code{"predict"} will compute performance measures for xtest using code{"type.measure"} from the cvfit object.
#' @param type Type of prediction required. Type '"link"' gives the linear predictors for '"binomial", '"multinomial"' or '"cox"' models; for '"gaussian"' models it gives the fitted values. Type '"response"' gives the fitted probabilities for '"binomial"' or '"multinomial"', and the fitted relative-risk for '"cox"'; for '"gaussian"' type '"response"' is equivalent to type '"link"'. Note that for '"binomial"' models, results are returned only for the class corresponding to the second level of the factor response. Type '"class"' applies only to '"binomial"' or '"multinomial"' models, and produces the class label corresponding to the maximum probability.
#' @param s Value of the penalty parameter 'lambda' at which predictions are required. Will use the same lambda for all models; can be a numeric value, '"lambda.min"' or '"lambda.1se"'. Default is '"lambda.min"'.
#' @param gamma For use only when 'relax = TRUE' was specified during training. Value of the penalty parameter 'gamma' at which predictions are required. Will use the same gamma for all models; can be a numeric value, '"gamma.min"' or '"gamma.1se"'. Default is '"gamma.min"'.
#' @param return.link If \code{TRUE}, will additionally return the linear link for the overall, pretrained and individual models: \code{linkoverall}, \code{linkpre} and \code{linkind}.
#' 
#' @return A list containing the requested predictions. If \code{ytest} is included, will also return error measures.
#' \item{call}{The call that produced this object.}
#' \item{alpha}{The value(s) of alpha used to generate predictions. Will be the same alpha used to in model training.}
#' \item{yhatoverall}{Predictions from the overall model.}
#' \item{yhatind}{Predictions from the individual models.}
#' \item{yhatpre}{Predictions from the pretrained models.}
#' \item{supoverall}{Indices of the features selected by the overall model.}
#' \item{supind}{Union of the indices of the features selected by the individual models.}
#' \item{suppre.common}{Features selected in the first stage of pretraining.}
#' \item{suppre.individual}{Union of the indices of the features selected by the pretrained models, without the features selected in the first stage.}
#' \item{type.measure}{If \code{ytest} is supplied, the string name of the computed performance measure.}
#' \item{erroverall}{If \code{ytest} is supplied, performance for the overall model. This is a named vector containing performance for (1) the entire dataset, (2) the average performance across groups, (3) the average performance across groups weighted by group size and (4) group-specific performance.}
#' \item{errind}{If \code{ytest} is supplied, performance for the overall model. As described in \code{erroverall}.}
#' \item{errpre}{If \code{ytest} is supplied, performance for the overall model. As described in \code{erroverall}.}
#' \item{linkoverall}{If\code{return.link} is TRUE, return the linear link from the overall model.}
#' \item{linkind}{If\code{return.link} is TRUE, return the linear link from the individual models.}
#' \item{linkpre}{If\code{return.link} is TRUE, return the linear link from the pretrained models.}
#'
#' @examples
#' # Gaussian example
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group
#'
#' outtest = gaussian.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#' 
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' pred = predict(fit, xtest, groupstest, ytest=ytest)
#' pred
#'
#' @import glmnet
#' @method predict ptLasso
#' @export
predict.ptLasso=function(fit, xtest, groupstest=NULL, ytest=NULL, 
                         type = c("link", "response", "class"),
                         s="lambda.min", gamma = "gamma.min", return.link=FALSE){

    this.call <- match.call()
    
    type = match.arg(type)
    family = fit$call$family
    type.measure = fit$call$type.measure
    group.intercepts = fit$call$group.intercepts

    original.groups = groupstest
    legend = fit$group.legend
    groupstest = transform.groups(groupstest, legend = legend)
    
    if(type == "class" & !(family %in% c("binomial", "multinomial")) ){
        stop("Class prediction is only valid for binomial and multinomial models.")
    }

    errFun = function(y, predmat){
            predmat.expanded = predmat
            dims = if(family == "multinomial") { dim(predmat) } else { length(predmat) }
            dim(predmat.expanded) <- c(dims, 1)
            as.numeric(assess.glmnet(predmat.expanded, newy=y, family=family)[type.measure])
         }

    if(!is.null(ytest)){
        if(family == "cox") {
            if(!check.Surv(ytest)) stop("For survival models, ytest must be a matrix with columns 'time' (>0) and 'status', or a Surv object.")
            if(!inherits(ytest, "Surv")) {
                if(is.null(colnames(ytest))) {
                    message("ytest missing column names; assuming column 1 is time and column 2 is status.")
                    colnames(ytest) = c("time", "status")
                }
            }
        }
    }

    
    if(fit$call$use.case=="inputGroups") out=predict.ptLasso.inputGroups(fit, xtest, groupstest=groupstest, ytest=ytest, errFun=errFun, type=type, call=this.call, family=family, type.measure=type.measure, s=s, gamma=gamma, return.link=return.link, group.intercepts=group.intercepts)
    if(fit$call$use.case=="targetGroups") out=predict.ptLasso.targetGroups(fit, xtest, ytest=ytest, type=type, call=this.call, family=family, type.measure=type.measure, s=s, gamma=gamma, return.link=return.link)
    class(out)="predict.ptLasso"
    return(out)
}

#' Predict function for input grouped data
#' @noRd
predict.ptLasso.inputGroups=function(fit, xtest, groupstest, errFun, family, type.measure, type, call, group.intercepts,
                                     s="lambda.min", gamma="gamma.min", return.link=FALSE, ytest=NULL){

    if(is.null(groupstest)) stop("Need group IDs for test data.")
        
    k=fit$k
    
    predgroups = sort(unique(groupstest))
    group.names = fit$group.legend[predgroups]

    onehot.test = NULL
    if(group.intercepts == TRUE){
        groupstest = factor(groupstest, levels=fit$group.levels)
        onehot.test = model.matrix(~groupstest - 1)
        if(family != "cox") onehot.test = onehot.test[, 2:k, drop=FALSE]
    }
    
    phatall = predict(fit$fitoverall, cbind(onehot.test, xtest), type="link", s=s, gamma=gamma) 

    yhatoverall = predict(fit$fitoverall, cbind(onehot.test, xtest), type=type, s=s, gamma=gamma) 
    
    if(!is.null(ytest)){
        if(family == "multinomial") {
            if(type.measure == "class"){
                erroverall=errFun(ytest, phatall[, , 1])
                erroverall.classes=sapply(predgroups, function(kk) errFun(ytest[groupstest == kk], phatall[groupstest == kk, , 1]))  
            } 
        } else if(family=="cox") {
            erroverall=errFun(ytest, as.numeric(phatall))
            erroverall.classes=sapply(predgroups, function(kk) errFun(ytest[groupstest == kk,], phatall[groupstest == kk]))
        } else {
            erroverall=errFun(ytest, as.numeric(phatall))
            erroverall.classes=sapply(predgroups, function(kk) errFun(ytest[groupstest == kk], phatall[groupstest == kk]))
        }
      }

    
    errpre=errind=rep(NA,k)
    
    if(family == "multinomial") ncolumns = length(fit$fitoverall$glmnet.fit$beta)
    if((family == "binomial") | (family == "gaussian") | (family == "cox")) ncolumns = 1
    phatpre=yhatpre=matrix(NA,nrow(xtest), ncolumns)
    phatind=yhatind=matrix(NA,nrow(xtest), ncolumns)

    if(type == "class"){
        yhatpre = yhatind = rep(NA, nrow(xtest))
    }
    
    # preTraining predictions
    for(kk in predgroups){
        test.ix = groupstest == kk
        
        # Pretraining predictions
        offsetTest = (1-fit$alpha) * predict(fit$fitoverall, cbind(onehot.test[test.ix, ], xtest[test.ix,]), s=fit$fitoverall.lambda, gamma=fit$fitoverall.gamma, type="link")
        if(family == "multinomial") offsetTest = offsetTest[, , 1]                             
        phatpre[test.ix, ] = predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type="link", s=s, gamma=gamma) 

        # Individual model predictions
        phatind[test.ix, ] = predict(fit$fitind[[kk]], xtest[test.ix,], type="link", s=s, gamma=gamma)  
        
        if(type == "class"){
            yhatpre[test.ix] = predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type=type, s=s, gamma=gamma) 
            yhatind[test.ix] = predict(fit$fitind[[kk]], xtest[test.ix,], type=type, s=s, gamma=gamma)
        } else {
            yhatpre[test.ix, ] = predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type=type, s=s, gamma=gamma)
            yhatind[test.ix, ] = predict(fit$fitind[[kk]], xtest[test.ix,], type=type, s=s, gamma=gamma) 
        }

        if(!is.null(ytest)){
            if( family == "multinomial" ){
                errpre[kk] = errFun(ytest[test.ix], phatpre[test.ix, ])
                errind[kk] = errFun(ytest[test.ix], phatind[test.ix, ])
            } else if(family == "cox")   {
                errpre[kk] = errFun(ytest[test.ix,], phatpre[test.ix, ])
                errind[kk] = errFun(ytest[test.ix,], phatind[test.ix, ,drop=FALSE] ) 
            } else {
                errpre[kk] = errFun(ytest[test.ix], phatpre[test.ix])
                errind[kk] = errFun(ytest[test.ix], phatind[test.ix]) 
            }
        }
    } 
    
    if(!is.null(ytest)){
        group.weights = table(as.numeric(groupstest))/length(groupstest)
        erroverall = c(erroverall,
                   mean( erroverall.classes),
                   weighted.mean( erroverall.classes, group.weights),
                   erroverall.classes)
        errind = c(errFun(ytest, phatind),
                   mean(errind, na.rm=TRUE),
                   weighted.mean(errind[!is.na(errind)], group.weights),
                   errind[!is.na(errind)])
        errpre = c(errFun(ytest, phatpre),
                   mean(errpre, na.rm=TRUE),
                   weighted.mean(errpre[!is.na(errpre)], group.weights),
                   errpre[!is.na(errpre)])
        names(erroverall) = names(errpre) = names(errind) = c("allGroups", "mean", "wtdMean", paste0("group_", group.names))
        
        if(fit$call$family == "gaussian") {
            erroverall = erroverall[which(names(erroverall) != "wtdMean")]
            errpre = errpre[which(names(errpre) != "wtdMean")]
            errind = errind[which(names(errind) != "wtdMean")]
            
            erroverall = c(erroverall, "r^2" = r2(yhatoverall, ytest))
            errpre = c(errpre, "r^2" = r2(yhatpre, ytest))
            errind = c(errind, "r^2" = r2(yhatind, ytest))
        }
    }

    suppre.common     = get.overall.support(fit, s=fit$fitoverall.lambda, gamma=gamma)
    suppre.individual = setdiff(get.pretrain.support(fit, s=s, gamma=gamma), suppre.common)
    
    out = enlist(yhatoverall = as.numeric(yhatoverall),
                 yhatind = as.numeric(yhatind),
                 yhatpre = as.numeric(yhatpre), 
                 suppre.common,
                 suppre.individual,
                 supind  = get.individual.support(fit, s=s, gamma=gamma),
                 supoverall  = get.overall.support(fit, s=s, gamma=gamma),
                 alpha = fit$alpha,
                 type.measure = type.measure,
                 call)

    if(return.link){
        out$linkoverall = phatall
        out$linkind = phatind
        out$linkpre = phatpre 
    }
    
    if(!is.null(ytest)) {
        out$erroverall = erroverall
        out$errind = errind
        out$errpre = errpre
    } 

    return(out)

}

#' Check survival object
#' @param y User entered response
#' @noRd
check.Surv <- function(y){
    if(inherits(y, "Surv")) return(TRUE)
    if(is.matrix(y) & ncol(y) == 2) return(TRUE)
    return(FALSE)
}

#' Compute r^2
#' @noRd
r2 <- function(yhat, y) 1 - sum((yhat - y)^2)/sum((y - mean(y))^2)

#' Call assess.glmnet for multinomial
#' @noRd
binomial.measure = function(newy, one.phat.column, measure = "deviance"){
     as.numeric(assess.glmnet(one.phat.column, newy=newy, family="binomial")[measure])
}

#' Predict function for target grouped data
#' @noRd
predict.ptLasso.targetGroups=function(fit, xtest, family, type.measure, type, call,
                                      return.link=FALSE, ytest=NULL, s="lambda.min",
                                      gamma="gamma.min", pred.groups = 1:fit$k, process.results=TRUE){
   
    k=fit$k
    groups=fit$y

    phatind = phatpre = matrix(NA, nrow=nrow(xtest), ncol=length(pred.groups))

    phatall = predict(fit$fitoverall, xtest, s=s,type="link")[, , 1] 
    
    offsetMatrix = (1-fit$alpha) * predict(fit$fitoverall, xtest, s=fit$fitoverall.lambda, type="link")[, , 1]
    
    ix = 1
    
    for(kk in pred.groups){
        
        # Pretraining predictions
        phatpre[,ix] = as.numeric(predict(fit$fitpre[[kk]], xtest, s=s, newoffset=offsetMatrix[, kk], type="link")) 

        # Individual predictions
        phatind[,ix] = as.numeric(predict(fit$fitind[[kk]], xtest, s=s, type="link"))

        ix = ix + 1
    }
    
    suppre.common = get.overall.support(fit, s=fit$fitoverall.lambda)
    if(process.results){
        suppre.individual = setdiff(get.pretrain.support(fit, s=s), suppre.common)
        
        return(process.targetGroup.results(phatall = phatall, phatpre = phatpre, phatind = phatind,
                                           ytest = ytest, k = k, type.measure = type.measure,
                                           return.link = return.link, type = type,
                                           alpha = fit$alpha, s = s, call = call, fit = fit,
                                           suppre.common = suppre.common,
                                           suppre.individual = suppre.individual))
    } else {
        suppre.individual = setdiff(get.pretrain.support(fit, s=s, groups=pred.groups, includeOverall=FALSE), suppre.common)
        return(enlist(phatall, phatpre, phatind, suppre.common, suppre.individual))
    }
}


process.targetGroup.results <- function(phatall, phatpre, phatind,
                                        ytest, k, type.measure,
                                        return.link, type,
                                        alpha, s, gamma, call, fit,
                                        suppre.common,
                                        suppre.individual
                                        ){
    multinomialErrFun = function(y, predmat){
            predmat.expanded = predmat
            dim(predmat.expanded) = c(dim(predmat), 1)
            as.numeric(assess.glmnet(predmat.expanded, newy=y, family="multinomial")[type.measure])
    }
    
    yhatind=yhatpre=matrix(NA, nrow=nrow(phatall), ncol=ncol(phatall))
    erroverall=errind=errpre=rep(NA, k)
    
    if(!is.null(ytest)){
        for(kk in 1:k){
            yytest = rep(0, length(ytest))
            yytest[ytest==kk]=1

            errind[kk] = binomial.measure(newy = yytest, one.phat.column = phatind[, kk], measure=type.measure)
            errpre[kk] = binomial.measure(newy = yytest, one.phat.column = phatpre[, kk], measure=type.measure)
        }
        errind = c(multinomialErrFun(ytest, phatind), mean(errind), errind)
        errpre = c(multinomialErrFun(ytest, phatpre), mean(errpre), errpre)
        erroverall = c(multinomialErrFun(ytest, phatall), NA, rep(NA, k))
        names(erroverall) = names(errpre) = names(errind) = c("overall", "mean", paste0("group_", as.character(1:k)))
     }
    
    yhatind = phatind; yhatpre = phatpre; yhatoverall = phatall
    if(type == "class") {
        yhatind = apply(yhatind, 1, which.max)
        yhatpre = apply(yhatpre, 1, which.max)
        yhatoverall = apply(yhatoverall, 1, which.max)
    }
    
    out = enlist(yhatoverall, yhatind, yhatpre,
                 suppre.individual,
                 suppre.common,
                 supind = get.individual.support(fit, s=s, gamma=gamma),
                 supoverall = get.overall.support(fit, s=s, gamma=gamma),
                 alpha = alpha,
                 type.measure = type.measure,
                 call)
    
    if(!is.null(ytest)) {
        out$errind = errind
        out$errpre = errpre
        out$erroverall = erroverall
    }

    if(return.link){
        out$linkoverall = phatall
        out$linkpre = phatpre
        out$linkind = phatind
    }
    
    class(out) = "predict.ptLasso"

    return(out)

}
