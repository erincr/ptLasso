
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
#' @param return.link If \code{TRUE}, will additionally return the linear link for the overall, pretrained and individual models: \code{linkall}, \code{linkpre} and \code{linkind}.
#' @return A list containing the requested predictions. If \code{ytest} is included, will also return error measures.
#' \item{call} The call that produced this object.
#' \item{alpha} The value(s) of alpha used to generate predictions.
#' \item{yhatall} Predictions from the overall model.
#' \item{yhatind} Predictions from the individual models.
#' \item{yhatpre} Predictions from the pretrained models.
#' \item{supall} Indices of the features selected by the overall model.
#' \item{supind} Union of the indices of the features selected by the individual models.
#' \item{suppre} Union of the indices of the features selected by the pretrained models. Includes features selected in the first stage of pretraining.
#' \item{type.measure} If \code{ytest} is supplied, the performance measure computed.
#' \item{errall} If \code{ytest} is supplied, performance for the overall model. This is a named vector containing performance for (1) the entire dataset, (2) the average performance across groups, (3) the average performance across groups weighted by group size and (4) group-specific performance.
#' \item{errind} If \code{ytest} is supplied, performance for the overall model. As described in \code{errall}.
#' \item{errpre} If \code{ytest} is supplied, performance for the overall model. As described in \code{errall}.
#' \item{linkall} If\code{return.link} is TRUE, return the linear link from the overall model.
#' \item{linkind} If\code{return.link} is TRUE, return the linear link from the individual models.
#' \item{linkpre} If\code{return.link} is TRUE, return the linear link from the pretrained models.
#'
#'
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
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
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.1se", alphatype = "varying", alpha = .3)
#' pred
#'
#' # Specify a vector of choices for alpha: 
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min", alphatype = "varying", alpha = c(.1, .2, .3, .4, .5))
#' pred
#'
#' @import glmnet
#' @method predict cv.ptLasso
#' @export
predict.cv.ptLasso=function(cvfit, xtest,  groupstest=NULL, ytest=NULL, alpha=NULL, alphatype = c("fixed", "varying"),
                            type = c("link", "response", "class"), s = "lambda.min", return.link = FALSE){

    if(missing("xtest")) stop("Please supply xtest.")

    this.call <- match.call()
    
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

        out = predict.ptLasso(fit, xtest, groupstest=groupstest, ytest=ytest, type = type, s = s)
        out$call = this.call
        
        return(out)
    } else {
        if(!all(sapply(alpha, function(x) any(abs(x - cvfit$errpre[, "alpha"]) < close.enough)))) stop("Includes at least one invalid choice of alpha. Please choose alpha from cvfit$alphalist.")
        if(length(alpha) != cvfit$fit[[1]]$k) stop("Must have one alpha for each group.")

        model.ix = sapply(alpha, function(a) which(abs(a - cvfit$errpre[, "alpha"]) < close.enough))
        model.ix = unname(model.ix)

        predgroups = sort(unique(groupstest))
        
        results = lapply(predgroups,
                         function(ix) {predict.ptLasso(cvfit$fit[[model.ix[ix]]],
                                                      xtest[groupstest == ix, ],
                                                      groupstest=groupstest[groupstest == ix],
                                                      ytest=ytest[groupstest == ix],
                                                      type = type, s = s, return.link=TRUE)}
                         )
        
        all.preds = pre.preds = ind.preds = rep(NA, nrow(xtest))
        all.link = pre.link = ind.link = rep(NA, nrow(xtest))
        for(kk in 1:length(predgroups)){
            all.preds[groupstest == predgroups[kk]] = results[[kk]]$yhatall
            pre.preds[groupstest == predgroups[kk]] = results[[kk]]$yhatpre
            ind.preds[groupstest == predgroups[kk]] = results[[kk]]$yhatind

            all.link[groupstest == predgroups[kk]] = results[[kk]]$linkall
            pre.link[groupstest == predgroups[kk]] = results[[kk]]$linkpre
            ind.link[groupstest == predgroups[kk]] = results[[kk]]$linkind
        }
        
        if(!is.null(ytest)){
            group.weights = table(groupstest)/length(groupstest)
            errall = sapply(results, function(r) r$errall[grepl("group", names(r$errall))])
            errall = c(as.numeric(assess.glmnet(all.link, newy=ytest, family=cvfit$family)[cvfit$type.measure]),
                       mean(errall),
                       weighted.mean(errall, w = group.weights),
                       errall)

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

            names(errall) = names(errpre) = names(errind) = c("overall", "mean", "wtdMean", paste0("group", 1:length(results)))
         }
      
        out = enlist(            
            yhatall = all.preds,
            yhatind = ind.preds, 
            yhatpre = pre.preds,

            suppre = sort(unique(unlist(lapply(results, function(x) x$suppre)))), #lapply(results, function(x) x$suppre),
            supall = results[[1]]$supall,
            supind = results[[1]]$supind,

            use.case = cvfit$fit[[1]]$call$use.case,

            type.measure = cvfit$fit[[1]]$call$type.measure,

            alpha,
            call = this.call
        )

        if(return.link){
            out$linkall = all.link
            out$linkind = ind.link
            out$linkpre = pre.link
        }
        
        if(!is.null(ytest)){
            out$errpre = errpre
            out$errind = errind
            out$errall = errall
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
#' @param s For use with 'fit.method = "glmnet"' only. Value of the penalty parameter 'lambda' at which predictions are required. Will use the same lambda for all models; can be a numeric value, '"lambda.min"' or '"lambda.1se"'. Default is '"lambda.min"'.
#' @param which For use with 'fit.method = "sparsenet"' only. Either the paramaters of the minimum of the CV curves (default "parms.min" or the parameters corresponding to the one standard-error rule "parms.1se"). Default is "parms.min".
#' @param return.link If \code{TRUE}, will additionally return the linear link for the overall, pretrained and individual models: \code{linkall}, \code{linkpre} and \code{linkind}.
#' 
#' @return A list containing the requested predictions. If \code{ytest} is included, will also return error measures.
#' \item{call} The call that produced this object.
#' \item{alpha} The value(s) of alpha used to generate predictions. Will be the same alpha used to in model training.
#' \item{yhatall} Predictions from the overall model.
#' \item{yhatind} Predictions from the individual models.
#' \item{yhatpre} Predictions from the pretrained models.
#' \item{supall} Indices of the features selected by the overall model.
#' \item{supind} Union of the indices of the features selected by the individual models.
#' \item{suppre} Union of the indices of the features selected by the pretrained models. Includes features selected in the first stage of pretraining.
#' \item{type.measure} If \code{ytest} is supplied, the performance measure computed.
#' \item{errall} If \code{ytest} is supplied, performance for the overall model. This is a named vector containing performance for (1) the entire dataset, (2) the average performance across groups, (3) the average performance across groups weighted by group size and (4) group-specific performance.
#' \item{errind} If \code{ytest} is supplied, performance for the overall model. As described in \code{errall}.
#' \item{errpre} If \code{ytest} is supplied, performance for the overall model. As described in \code{errall}.
#' \item{linkall} If\code{return.link} is TRUE, return the linear link from the overall model.
#' \item{linkind} If\code{return.link} is TRUE, return the linear link from the individual models.
#' \item{linkpre} If\code{return.link} is TRUE, return the linear link from the pretrained models.
#'
#' @examples
#' # Gaussian example
#' 
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group
#'
#' # Test data
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
                         type = c("link", "response", "class"), s="lambda.min", which="parms.min", return.link=FALSE){

    this.call <- match.call()
    
    type = match.arg(type)
    family = fit$call$family
    type.measure = fit$call$type.measure
    
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
            if(!is.matrix(ytest) | (is.matrix(ytest) & ncol(ytest) != 2)) stop("For survival models, ytest must be a matrix with columns 'time' (>0) and 'status', or a Surv object.")
            if(!("Surv" %in% class(ytest))) {
                if(is.null(colnames(ytest))) {
                    message("ytest missing column names; assuming column 1 is time and column 2 is status.")
                    colnames(ytest) = c("time", "status")
                }
            }
        }
    }
    
    if(fit$call$use.case=="inputGroups") out=predict.ptLasso.inputGroups(fit, xtest, groupstest=groupstest, ytest=ytest, errFun=errFun, type=type, call=this.call, family=family, type.measure=type.measure, s=s, which=which, return.link=return.link)
    if(fit$call$use.case=="targetGroups") out=predict.ptLasso.targetGroups(fit, xtest, ytest=ytest, errFun=errFun, type=type, call=this.call, family=family, type.measure=type.measure, s=s, return.link=return.link)
    class(out)="predict.ptLasso"
    return(out)
}

#' Predict function for input grouped data
#' @noRd
predict.ptLasso.inputGroups=function(fit, xtest, groupstest, errFun, family, type.measure, type, call, return.link=FALSE, ytest=NULL, s="lambda.min", which="parms.min"){

    if(inherits(fit$fitall, "cv.sparsenet")) this.predict = function(...) {
        params = list(...)
        params$which = which
        params$type = "response"
        do.call(predict, params)
     }
    if(inherits(fit$fitall, "cv.glmnet")) this.predict = function(...) predict(..., s=s) 
    
    if(is.null(groupstest)) stop("Need group IDs for test data.")
        
    k=fit$k
    
    predgroups = sort(unique(groupstest))
    
    groupstest = factor(groupstest, levels=fit$group.levels)
    onehot.test = model.matrix(~groupstest - 1)
    if(family != "cox") onehot.test = onehot.test[, 2:k, drop=FALSE]
    
    phatall = this.predict(fit$fitall, cbind(onehot.test, xtest), type="link") 

    yhatall = this.predict(fit$fitall, cbind(onehot.test, xtest), type=type) 
    
    if(!is.null(ytest)){
        if(family == "multinomial") {
            if(type.measure == "class"){
                errall=errFun(ytest, phatall[, , 1])
                errall.classes=sapply(predgroups, function(kk) errFun(ytest[groupstest == kk], phatall[groupstest == kk, , 1]))  
            } 
        } else if(family=="cox") {
            errall=errFun(ytest, as.numeric(phatall))
            errall.classes=sapply(predgroups, function(kk) errFun(ytest[groupstest == kk,], phatall[groupstest == kk]))
        } else {
            errall=errFun(ytest, as.numeric(phatall))
            errall.classes=sapply(predgroups, function(kk) errFun(ytest[groupstest == kk], phatall[groupstest == kk]))
        }
      }

    
    errpre=errind=rep(NA,k)
    
    if(family == "multinomial") ncolumns = length(fit$fitall$glmnet.fit$beta)
    if((family == "binomial") | (family == "gaussian") | (family == "cox")) ncolumns = 1
    phatpre=yhatpre=matrix(NA,nrow(xtest), ncolumns)
    phatind=yhatind=matrix(NA,nrow(xtest), ncolumns)

    if(type == "class"){
        yhatpre = yhatind = rep(NA, nrow(xtest))
    }
    
    errallInd=NULL
    
    # individual group//class predictions
    for(kk in predgroups){
     
        test.ix  = groupstest == kk

        phatind[test.ix, ] = this.predict(fit$fitind[[kk]], xtest[test.ix,], type="link") 

        if(type == "class"){
            yhatind[test.ix] = this.predict(fit$fitind[[kk]], xtest[test.ix,], type=type)
        } else {
            yhatind[test.ix, ] = this.predict(fit$fitind[[kk]], xtest[test.ix,], type=type) 
        }

        if(!is.null(ytest)){
            if( family == "multinomial" ){
                errind[kk]=errFun(ytest[test.ix], phatind[test.ix, ])
            } else if(family == "cox")   {
                errind[kk]=errFun(ytest[test.ix,], phatind[test.ix, ,drop=FALSE] ) 
            } else {
                errind[kk]=errFun(ytest[test.ix], phatind[test.ix]) 
            }
        }

    }
    
    if(!is.null(ytest)) errind.overall=errFun(ytest, phatind)
 
    # preTraining predictions
    for(kk in predgroups){
        test.ix = groupstest == kk
        
        # Pretraining predictions
        if(inherits(fit$fitall, "cv.sparsenet")){
            offsetTest = (1-fit$alpha) * predict(fit$fitall, cbind(onehot.test[test.ix, ], xtest[test.ix,]), which=fit$fitall.which, type="response")
        } else {
            offsetTest = (1-fit$alpha) * predict(fit$fitall, cbind(onehot.test[test.ix, ], xtest[test.ix,]), s=fit$fitall.lambda, type="link")
        }
        if(family == "multinomial") offsetTest = offsetTest[, , 1]                             
        phatpre[test.ix, ] = this.predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest,type="link") 

        # Individual model predictions
        phatind[test.ix, ] = this.predict(fit$fitind[[kk]], xtest[test.ix,], type="link")  
        
        if(type == "class"){
            yhatpre[test.ix] = this.predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type=type) 
            yhatind[test.ix] = this.predict(fit$fitind[[kk]], xtest[test.ix,], type=type)
        } else {
            yhatpre[test.ix, ] = this.predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type=type)
            yhatind[test.ix, ] = this.predict(fit$fitind[[kk]], xtest[test.ix,], type=type) 
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
        errall = c(errall,
                   mean( errall.classes),
                   weighted.mean( errall.classes, group.weights),
                   errall.classes)
        errind = c(errind.overall,
                   mean(errind, na.rm=TRUE),
                   weighted.mean(errind[!is.na(errind)], group.weights),
                   errind[!is.na(errind)])
        errpre = c(errFun(ytest, phatpre),
                   mean(errpre, na.rm=TRUE),
                   weighted.mean(errpre[!is.na(errpre)], group.weights),
                   errpre[!is.na(errpre)])
        names(errall) = names(errpre) = names(errind) = c("overall", "mean", "wtdMean", paste0("group", sort(unique(groupstest))))
    }

    out = enlist(yhatall = as.numeric(yhatall),
                 yhatind = as.numeric(yhatind),
                 yhatpre = as.numeric(yhatpre), 
                 suppre  = get.pretrain.support(fit, s=s, which=which),
                 supind  = get.individual.support(fit, s=s, which=which),
                 supall  = get.overall.support(fit, s=s, which=which),
                 alpha = fit$alpha,
                 type.measure = type.measure,
                 call)

    if(return.link){
        out$linkall = phatall
        out$linkind = phatind
        out$linkpre = phatpre 
    }
    
    if(!is.null(ytest)) {
        out$errall = errall
        out$errind = errind
        out$errpre = errpre
    } 

    return(out)

}


binomial.measure = function(newy, one.phat.column, measure = "deviance"){
     as.numeric(assess.glmnet(one.phat.column, newy=newy, family="binomial")[measure])
}

#' Predict function for target grouped data
#' @noRd
predict.ptLasso.targetGroups=function(fit, xtest,  errFun, family, type.measure, type, call, return.link=FALSE, ytest=NULL, s="lambda.min"){
   
    k=fit$k
    groups=fit$y

    phatind=phatpre=matrix(NA, nrow=nrow(xtest), ncol=k)
    yhatind=yhatpre=matrix(NA, nrow=nrow(xtest), ncol=k)
    errall=errind=errpre=rep(NA, k) 

    phatall = predict(fit$fitall, xtest, s=s,type="link")[, , 1] 
    yhatall = predict(fit$fitall, xtest, s=s,type=type)
    yhatall = if(type == "class"){ as.numeric(yhatall[, 1]) } else { yhatall[, , 1] }
    
    if(!is.null(ytest)){
        errall.overall = errFun(ytest, phatall)
    }
   
    offsetMatrix = (1-fit$alpha) * predict(fit$fitall, xtest, s=fit$fitall.lambda, type="link")[, , 1]
    for(kk in 1:k){
        # Pretraining predictions
        offsetTest = offsetMatrix[, kk]
        phatpre[,kk] = as.numeric(predict(fit$fitpre[[kk]], xtest, s=s, newoffset=offsetTest, type="link")) 

        # Individual predictions
        phatind[,kk] = as.numeric(predict(fit$fitind[[kk]],xtest,s=s, type="link"))
        
        if(!is.null(ytest)){
            yytest = rep(0, length(ytest))
            yytest[ytest==kk]=1
         
            errind[kk] = binomial.measure(newy = yytest, one.phat.column = phatind[, kk], measure=type.measure)
            errpre[kk] = binomial.measure(newy = yytest, one.phat.column = phatpre[, kk], measure=type.measure)
        }
    }
    
    if(!is.null(ytest)){
        errind = c(errFun(ytest, phatind), mean(errind), errind)
        errpre = c(errFun(ytest, phatpre), mean(errpre), errpre)
        errall = c(errall.overall, NA, rep(NA, k))
        names(errall) = names(errpre) = names(errind) = c("overall", "mean", paste0("group", as.character(1:k)))
     }

    
    yhatind = phatind; yhatpre = phatpre
    if(type == "class") {
        yhatind = apply(yhatind, 1, which.max)
        yhatpre = apply(yhatpre, 1, which.max)
    }

    out = enlist(yhatall, yhatind, yhatpre,
                 suppre = get.pretrain.support(fit, s=s),
                 supind = get.individual.support(fit, s=s),
                 supall = get.overall.support(fit, s=s),
                 alpha = fit$alpha,
                 type.measure = type.measure,
                 call)
    
    if(!is.null(ytest)) {
        out$errind = errind
        out$errpre = errpre
        out$errall = errall
    }

    if(return.link){
        out$linkall = phatall
        out$linkpre = phatpre
        out$linkind = phatind
    }
    
    class(out) = "predict.ptLasso"
    return(out)
}
