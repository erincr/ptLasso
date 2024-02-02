#
#' Predict using a cv.ptLasso object. 
#'
#' Return predictions and performance measures for a test set. 
#' 
#' @aliases predict.cv.ptLasso
#' @param cvfit Fitted \code{"cv.ptLasso"} object.
#' @param xtest Input matrix, matching the form used by \code{"cv.ptLasso"} for model training.
#' @param groupstest  A vector indicating to which group each observation belongs. Coding should match that used for model training. Will be NULL for target grouped data.
#' @param ytest Response variable. Optional. If included, \code{"predict"} will compute performance measures for xtest using code{"type.measure"} from the cvfit object.
#' @param alpha The chosen alpha to use for prediction. If NULL, this will rely on the choice of "alphatype".
#' @param alphatype Choice of '"fixed"' or '"varying"'. If '"fixed"', use the alpha that achieved best cross-validated performance. If '"varying"', each group uses the alpha that optimized the group-specific cross-validated performance.
#' @param type Type of prediction required. Type '"link"' gives the linear predictors for '"binomial", '"multinomial"' or '"cox"' models; for '"gaussian"' models it gives the fitted values. Type '"response"' gives the fitted probabilities for '"binomial"' or '"multinomial"', and the fitted relative-risk for '"cox"'; for '"gaussian"' type '"response"' is equivalent to type '"link"'. Note that for '"binomial"' models, results are returned only for the class corresponding to the second level of the factor response. Type '"class"' applies only to '"binomial"' or '"multinomial"' models, and produces the class label corresponding to the maximum probability.
#' @param s Value of the penalty parameter 'lambda' at which predictions are required. Will use the same lambda for all models; can be a numeric value, '"lambda.min"' or '"lambda.1se"'. Default is '"lambda.min"'.
#' @return A ...
#'
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#'
#' @import glmnet
#' @method predict cv.ptLasso
#' @export
predict.cv.ptLasso=function(cvfit, xtest,  groupstest=NULL, ytest=NULL, alpha=NULL, alphatype = c("fixed", "varying"),
                            type = c("link", "response", "class"), s = "lambda.min"){

    if(missing("xtest")) stop("Please supply xtest.")
    
    alphatype = match.arg(alphatype)
    if(is.null(alpha)) {
        if(alphatype == "fixed"){
            alpha = cvfit$alphahat
        } else if(alphatype == "varying") {
            alpha = cvfit$varying.alphahat
       }
    }
    
    if(length(alpha) == 1){
        if(!(alpha %in% cvfit$errpre[, "alpha"])) stop("Not a valid choice of alpha. Please choose alpha from cvfit$alphalist.")
        
        model.ix = which(cvfit$errpre[, "alpha"] == alpha)
        
        if(length(model.ix) > 1) model.ix = model.ix[1]
        
        fit = cvfit$fit[[model.ix]]

        return(predict.ptLasso(fit, xtest, groupstest=groupstest, ytest=ytest, type = type, s = s))
    } else {
        if(!all(sapply(alpha, function(x) x %in% cvfit$errpre[, "alpha"]))) stop("Includes at least one invalid choice of alpha. Please choose alpha from cvfit$alphalist.")
        if(length(alpha) != cvfit$fit[[1]]$k) stop("Must have one alpha for each group.")
        
        model.ix = sapply(alpha, function(a) which(a == cvfit$errpre[, "alpha"]))
        model.ix = unname(model.ix)
        
        results = lapply(1:length(model.ix),
                         function(ix) {predict.ptLasso(cvfit$fit[[model.ix[ix]]],
                                                      xtest[groupstest == ix, ],
                                                      groupstest=groupstest[groupstest == ix],
                                                      ytest=ytest[groupstest == ix],
                                                      type = type, s = s)}
                         )
        
        all.preds = pre.preds = ind.preds = rep(NA, nrow(xtest))
        all.link = pre.link = ind.link = rep(NA, nrow(xtest))
        for(kk in 1:length(model.ix)){
            all.preds[groupstest == kk] = results[[kk]]$yhatall
            pre.preds[groupstest == kk] = results[[kk]]$yhatpre
            ind.preds[groupstest == kk] = results[[kk]]$yhatind

            all.link[groupstest == kk] = results[[kk]]$linkall
            pre.link[groupstest == kk] = results[[kk]]$linkpre
            ind.link[groupstest == kk] = results[[kk]]$linkind
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
            yhatind = pre.preds,
            yhatpre = ind.preds,

            suppre = sort(unique(unlist(lapply(results, function(x) x$suppre)))), #lapply(results, function(x) x$suppre),
            supall = results[[1]]$supall,
            supind = results[[1]]$supind,

            use.case = cvfit$fit[[1]]$call$use.case,

            type.measure = cvfit$fit[[1]]$type.measure,

            alpha,
            call
        )
        
        if(!is.null(ytest)){
            out$errpre = errpre
            out$errind = errind
            out$errall = errall
        }
        
        class(out) = "predict.cv.ptLasso"
        
        return(out)
    }

    

}

#
#' Predict using a ptLasso object. 
#'
#' Return predictions and performance measures for a test set. 
#' 
#' @aliases predict.ptLasso
#' @param cvfit Fitted \code{"ptLasso"} object.
#' @param xtest Input matrix, matching the form used by \code{"ptLasso"} for model training.
#' @param groupstest  A vector indicating to which group each observation belongs. Coding should match that used for model training. Will be NULL for target grouped data.
#' @param type Type of prediction required. Type '"link"' gives the linear predictors for '"binomial", '"multinomial"' or '"cox"' models; for '"gaussian"' models it gives the fitted values. Type '"response"' gives the fitted probabilities for '"binomial"' or '"multinomial"', and the fitted relative-risk for '"cox"'; for '"gaussian"' type '"response"' is equivalent to type '"link"'. Note that for '"binomial"' models, results are returned only for the class corresponding to the second level of the factor response. Type '"class"' applies only to '"binomial"' or '"multinomial"' models, and produces the class label corresponding to the maximum probability.
#' @param s Value of the penalty parameter 'lambda' at which predictions are required. Will use the same lambda for all models; can be a numeric value, '"lambda.min"' or '"lambda.1se"'. Default is '"lambda.min"'.
#' @return A ..
#' @examples
#'
#' @import glmnet
#' @method predict ptLasso
#' @export
predict.ptLasso=function(fit, xtest, groupstest=NULL, ytest=NULL, offset = NULL,
                         type = c("link", "response", "class"), s="lambda.min"){

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
    
    if(fit$call$use.case=="inputGroups") out=predict.ptLasso.inputGroups(fit, xtest, groupstest=groupstest, ytest=ytest, errFun=errFun, type=type, call=this.call, family=family, type.measure=type.measure, s=s)
    if(fit$call$use.case=="targetGroups") out=predict.ptLasso.targetGroups(fit, xtest, ytest=ytest, errFun=errFun, type=type, call=this.call, family=family, type.measure=type.measure, s=s)
    class(out)="predict.ptLasso"
    return(out)
}

#' Predict function for input grouped data
#' @noRd
predict.ptLasso.inputGroups=function(fit, xtest, groupstest, errFun, family, type.measure, type, call, ytest=NULL, s="lambda.min"){

    if(is.null(groupstest)) stop("Need group IDs for test data.")
    
    predgroups = sort(unique(groupstest))
    
    groupstest = factor(groupstest, levels=fit$group.levels)
    onehot.test = model.matrix(~groupstest - 1)
    
    ytest.mean= 0
    if(family=="gaussian") ytest.mean = fit$y.mean
    
    k=fit$k
    
    phatall = predict(fit$fitall, cbind(onehot.test, xtest), type="link", s=s) 
    if(family=="gaussian")  phatall = phatall+ytest.mean

    yhatall = predict(fit$fitall, cbind(onehot.test, xtest), type=type, s=s) 
    if(family=="gaussian")  yhatall = yhatall+ytest.mean

    
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
            #browser()
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

        phatind[test.ix, ] = predict(fit$fitind[[kk]], xtest[test.ix,], type="link", s=s) 
        if(family=="gaussian")  phatind[test.ix, ] = phatind[test.ix, ] + ytest.mean

        if(type == "class"){
            yhatind[test.ix] = predict(fit$fitind[[kk]], xtest[test.ix,], type=type, s=s)
        } else {
            yhatind[test.ix, ] = predict(fit$fitind[[kk]], xtest[test.ix,], type=type, s=s) 
            if(family=="gaussian")  yhatind[test.ix, ] = yhatind[test.ix, ] + ytest.mean
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
        offsetTest = (1-fit$alpha) * predict(fit$fitall, cbind(onehot.test[test.ix, ], xtest[test.ix,]), s=fit$lamhat, type="link") 
        if(family == "multinomial") offsetTest = offsetTest[, , 1]                             
        phatpre[test.ix, ] = predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest,type="link", s=s) 

        # Individual model predictions
        phatind[test.ix, ] = predict(fit$fitind[[kk]], xtest[test.ix,], type="link", s=s)  
        
        if(type == "class"){
            yhatpre[test.ix] = predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type=type, s=s) 
            yhatind[test.ix] = predict(fit$fitind[[kk]], xtest[test.ix,], type=type, s=s)
        } else {
            yhatpre[test.ix, ] = predict(fit$fitpre[[kk]], xtest[test.ix,], newoffset=offsetTest, type=type, s=s)
            yhatind[test.ix, ] = predict(fit$fitind[[kk]], xtest[test.ix,], type=type, s=s) 
            if(family=="gaussian") {
                phatpre[test.ix, ] = phatpre[test.ix, ] + ytest.mean
                phatind[test.ix, ] = phatind[test.ix, ] + ytest.mean

                yhatpre[test.ix, ] = yhatpre[test.ix, ] + ytest.mean
                yhatind[test.ix, ] = yhatind[test.ix, ] + ytest.mean
            }
        }

        if(!is.null(ytest)){
            if( family == "multinomial" ){
                errpre[kk]=errFun(ytest[test.ix], phatpre[test.ix, ])
                errind[kk]=errFun(ytest[test.ix], phatind[test.ix, ])
            } else if(family == "cox")   {
                errpre[kk] = errFun(ytest[test.ix,], phatpre[test.ix, ])
                errind[kk]=errFun(ytest[test.ix,], phatind[test.ix, ,drop=FALSE] ) 
            } else {
                errpre[kk] = errFun(ytest[test.ix], phatpre[test.ix])
                errind[kk]=errFun(ytest[test.ix], phatind[test.ix]) 
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

    out = enlist(linkall = phatall, linkind = phatind, linkpre = phatpre,
                 yhatall = as.numeric(yhatall),
                 yhatind = as.numeric(yhatind),
                 yhatpre = as.numeric(yhatpre), 
                 suppre  = get.pretrain.support(fit, s=s),
                 supind  = get.individual.support(fit, s=s),
                 supall  = get.overall.support(fit, s=s),
                 alpha = fit$alpha,
                 type.measure = type.measure,
                 call)
    
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
predict.ptLasso.targetGroups=function(fit, xtest,  errFun, family, type.measure, type, call, ytest=NULL, s="lambda.min"){
   
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
   
    offsetMatrix = (1-fit$alpha) * predict(fit$fitall, xtest, s=fit$lamhat, type="link")[, , 1]
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

    if(!is.null(ytest)) {
        out = enlist(yhatall, yhatind, yhatpre,
                     errind, errpre, errall,
                     suppre = get.pretrain.support(fit, s=s),
                     supind = get.individual.support(fit, s=s),
                     supall = get.overall.support(fit, s=s),
                     alpha = fit$alpha,
                     type.measure = type.measure,
                     call)
    } else {
        out = enlist(yhatall, yhatind, yhatpre,
                     suppre = get.pretrain.support(fit, s=s),
                     supind = get.individual.support(fit, s=s),
                     supall = get.overall.support(fit, s=s),
                     alpha = fit$alpha,
                     type.measure = type.measure,
                     call)
    }
    class(out) = "predict.ptLasso"
    return(out)
}
