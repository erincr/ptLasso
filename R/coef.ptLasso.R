#' Get the support for pretrained models
#' 
#' Get the indices of nonzero coefficients from the pretrained models in a fitted ptLasso or cv.ptLasso object, excluding the intercept.
#'
#' @param fit fitted \code{"ptLasso"} or \code{"cv.ptLasso"} object.
#' @param s the choice of lambda to use. May be "lambda.min", "lambda.1se" or a numeric value. Default is "lambda.min".
#' @param gamma for use only when 'relax = TRUE' was specified during training. The choice of 'gamma' to use. May be "gamma.min" or "gamma.1se". Default is "gamma.min".
#' @param includeOverall whether to return the features that are chosen by the overall model and not the group-specific models (TRUE) or the features that are chosen by the overall model or the group-specific models (FALSE). Default is TRUE. Not used when 'use.case = "timeSeries"'.
#' @param commonOnly whether to return the features that are chosen by more than half of the group- or response-specific models (TRUE) or the features that are chosen by any of the group-specific models (FALSE). Default is FALSE.
#' @param groups which groups or responses to include when computing the support. Default is to include all groups/responses. 
#' @seealso \code{ptLasso}, \code{cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' # Train data
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#'
#' get.pretrain.support(fit) 
#' 
#' # only return features common to all groups 
#' get.pretrain.support(fit, commonOnly = TRUE) 
#' 
#' # group 1 only, don't include the overall model support
#' get.pretrain.support(fit, groups = 1, includeOverall = FALSE) 
#' 
#' # group 1 only, include the overall model support
#' get.pretrain.support(fit, groups = 1, includeOverall = TRUE) 
#' 
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' 
#' get.pretrain.support(cvfit)
#' get.pretrain.support(cvfit, groups = 1) 
#'  
#' @return If a ptLasso object is supplied, this returns a vector containing the indices of nonzero coefficients (excluding the intercept). If a cv.ptLasso object is supplied, this returns a list of results - one for each value of alpha. 
#'
#' @export
#'
get.pretrain.support <- function(fit, s="lambda.min", gamma="gamma.min", commonOnly = FALSE, includeOverall = TRUE, groups = 1:length(fit$fitind)) {
    if(inherits(fit, "cv.ptLasso")) return(lapply(fit$fit, function(m) get.pretrain.or.individual.support(m, s=s, gamma=gamma, commonOnly=commonOnly, includeOverall=includeOverall, groups = groups, model="pretrain"))) 
    get.pretrain.or.individual.support(fit, s=s, gamma=gamma, commonOnly=commonOnly, includeOverall=includeOverall, groups = groups, model="pretrain")
}

#' Get the support for individual models
#' 
#' Get the indices of nonzero coefficients from the individual models in a fitted ptLasso or cv.ptLasso object, excluding the intercept.
#'
#' @param fit fitted \code{"ptLasso"} or \code{"cv.ptLasso"} object.
#' @param s the choice of lambda to use. May be "lambda.min", "lambda.1se" or a numeric value. Default is "lambda.min".
#' @param gamma for use only when 'relax = TRUE' was specified during training. The choice of 'gamma' to use. May be "gamma.min" or "gamma.1se". Default is "gamma.min".
#' @param commonOnly whether to return the features that are chosen by more than half of the group- or response-specific models (TRUE) or the features that are chosen by any of the group-specific models (FALSE). Default is FALSE.
#' @param groups which groups or responses to include when computing the support. Default is to include all groups/responses. 
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' # Train data
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#'
#' get.individual.support(fit) 
#' 
#' # only return features common to all groups 
#' get.individual.support(fit, commonOnly = TRUE) 
#' 
#' # group 1 only
#' get.individual.support(fit, groups = 1) 
#' 
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' 
#' get.individual.support(cvfit)
#' 
#' # group 1 only
#' get.individual.support(cvfit, groups = 1) 
#'  
#' @return This returns a vector containing the indices of nonzero coefficients (excluding the intercept). 
#'
#' @export
#' @importFrom stats coef
#'
get.individual.support <- function(fit, s="lambda.min", gamma="gamma.min", commonOnly = FALSE, groups = 1:length(fit$fitind)) {
    if(inherits(fit, "cv.ptLasso")) return(get.pretrain.or.individual.support(fit$fit[[1]], s=s, gamma=gamma, commonOnly=commonOnly, groups = groups, model="individual")) 
    return(get.pretrain.or.individual.support(fit, s=s, gamma=gamma, commonOnly=commonOnly, groups = groups, model="individual"))
}

#' Helper function to get support for pretrained or individual models.
#' @noRd
get.pretrain.or.individual.support <- function(fit, s="lambda.min", gamma="gamma.min", model="pretrain", groups = groups, includeOverall=TRUE, commonOnly = FALSE){
    
    include.these = c()
    if((model == "pretrain") && (includeOverall == TRUE) & (fit$call$use.case != "timeSeries")){
        if(fit$alpha < 1) {
            if("fitoverall.gamma" %in% names(fit)){
                include.these = get.overall.support(fit, s=fit$fitoverall.lambda, gamma=fit$fitoverall.gamma)
            } else {
                include.these = get.overall.support(fit, s=fit$fitoverall.lambda)
            }
        }
    }

    model = if(model == "pretrain") {fit$fitpre} else {fit$fitind}
    
    suppre=vector("list",length(groups))
    bhatpre=vector("list",length(groups))
    
    ix = 1
    for(kk in groups){
        if(fit$call$use.case == "inputGroups"){
            if(fit$call$family == "multinomial"){
                bhatpre[[ix]] = coef(model[[kk]], s=s, gamma=gamma, exact=FALSE)
                bhatpre[[ix]] = do.call(cbind, bhatpre[[ix]])[-1, ]
                
                suppre[[ix]] = which(apply(bhatpre[[ix]], 1, function(x) sum(x != 0) > 0))
                suppre[[ix]] = sort(unique(c(suppre[[ix]], include.these)))
            } else {
                if(fit$call$family=="cox"){
                    bhatpre[[ix]] = as.numeric(coef(model[[kk]], s=s, gamma=gamma, exact=FALSE))
                } else {
                    bhatpre[[ix]] = as.numeric(coef(model[[kk]], s=s, gamma=gamma, exact=FALSE)[-1])
                }
                suppre[[ix]]=sort(unique(c(which(bhatpre[[ix]]!=0), include.these)))
            }
        } else if(fit$call$use.case == "targetGroups"){
            # This should always be a binomial (one vs. rest) model:
            bhatpre[[ix]] = as.numeric(coef(model[[kk]], s=s, gamma=gamma, exact=FALSE)[-1])
            suppre[[ix]] = sort(unique(c(which(bhatpre[[ix]]!=0), include.these)))
        } else if(fit$call$use.case %in% c("multiresponse", "timeSeries")){
            # This should always be gaussian (or logistic for timeSeries):
            bhatpre[[ix]] = as.numeric(coef(model[[kk]], s=s, gamma=gamma, exact=FALSE)[-1])
            suppre[[ix]] = sort(unique(c(which(bhatpre[[ix]]!=0), include.these)))
        } 
      
      ix = ix + 1
    }

    all.selected = sort(unique(unlist(suppre)))
    if(!commonOnly) return(all.selected)

    counts = sapply(all.selected, function(coeff) sum(sapply(suppre, function(supp) coeff %in% supp)))
    return(all.selected[counts > length(groups)/2])
}

#' Get the support for the overall model
#' 
#' Get the indices of nonzero coefficients from the overall model in a fitted ptLasso or cv.ptLasso object, excluding the intercept.
#'
#' @param fit fitted \code{"ptLasso"} or \code{"cv.ptLasso"} object.
#' @param s the choice of lambda to use. May be "lambda.min", "lambda.1se" or a numeric value. Default is "lambda.min".
#' @param gamma for use only when 'relax = TRUE' was specified during training. The choice of 'gamma' to use. May be "gamma.min" or "gamma.1se". Default is "gamma.min".
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' # Train data
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#'
#' get.overall.support(fit, s="lambda.min") 
#' get.overall.support(fit, s="lambda.1se") 
#' 
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' 
#' get.overall.support(cvfit, s="lambda.min") 
#' get.overall.support(cvfit, s="lambda.1se") 
#'  
#' @return This returns a vector containing the indices of nonzero coefficients (excluding the intercept). 
#'
#' @export
#' @importFrom stats coef
#'
get.overall.support <- function(fit, s="lambda.min", gamma="gamma.min"){

    if(fit$call$use.case == "timeSeries") stop("Overall support is not available for the time series use case. There is no overall model.")
    
    if(inherits(fit, "cv.ptLasso")) {
        if(is.null(s)) s = fit$fitoverall.lambda
        if("fitoverall.gamma" %in% names(fit)){
            if(is.null(gamma)) gamma = fit$fitoverall.gamma
        }
        return(get.overall.support(fit$fit[[1]], s = s, gamma = gamma))
    }

    if(is.null(s)) stop("s cannot be null.")
    
    coefs = coef(fit$fitoverall, s=s, gamma = gamma)
    k = fit$k
    
    
    # multinomial
    if(is.list(coefs)){
        if(fit$call$use.case == "inputGroups"){
            if(fit$call$group.intercepts == TRUE){
                return(sort(unique(unlist(lapply(coefs, function(cc) which(cc[-(1:k)] != 0)))))) # first k are group indicators
            } else {
                return(sort(unique(unlist(lapply(coefs, function(cc) which(cc[-1] != 0)))))) # no group indicators
            }
        } else if(fit$call$use.case == "targetGroups") {
            return(sort(unique(unlist(lapply(coefs, function(cc) which(cc[-1] != 0)))))) # no group indicators, only an intercept
        }
    }

    # other
    if(fit$call$use.case == "inputGroups"){
        if(fit$call$group.intercepts == TRUE){
            return(which(coefs[-(1:k)] != 0)) # first is intercept, next k-1 are group indicators -- if Cox, no intercept and k group indicators
        } else {
            return(which(coefs[-1] != 0)) # no group indicators
        }
    } else if(fit$call$use.case == "multiresponse"){
        return(which(rowSums(do.call(cbind, coefs)[-1, ]) != 0))
    }
}


#' Get the coefficients from a fitted ptLasso model.
#'
#' @aliases coef.ptLasso 
#' @param object fitted \code{"ptLasso"} object.
#' @param model string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".
#' @param \dots other arguments to be passed to the \code{"coef"} function. May be e.g. \code{s = "lambda.min"}.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#' # Train data
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' # Get all model coefficients.
#' names(coef(fit))
#'
#' coef(fit, model = "overall") # Overall model only
#' length(coef(fit, model = "individual")) # List of coefficients for each group model
#' length(coef(fit, model = "pretrain")) # List of coefficients for each group model
#'
#' @return Model coefficients. If \code{model = "overall"}, this function returns the output of \code{coef}. If \code{model} is "individual" or "pretrain", this function returns a list containing the results of \code{coef} for each group-specific model. If \code{model = "all"}, this returns a list containing all (overall, individual and pretrain) coefficients. 
#'
#' @method coef ptLasso
#' @export
#' @importFrom stats coef
#'
coef.ptLasso=function(object, model = c("all", "individual", "overall", "pretrain"), ...){
    model = match.arg(model)
    is.ts = object$call$use.case == "timeSeries"
    
    if( (model == "overall") && (is.ts) ) stop("There is no overall model for time series data")

    if((model == "all") | (model == "individual")) individual = lapply(object$fitind, function(x) coef(x, ...))
    if((model == "all") | (model == "pretrain"))   pretrain   = lapply(object$fitpre, function(x) coef(x, ...))
    if(!is.ts) {
        if((model == "all") | (model == "overall"))    overall    = coef(object$fitoverall, ...)
    }

    if(model == "all")  {
        res = list( individual = individual,  pretrain = pretrain )
        if(is.ts) return(res)
        res$overall = overall
        return(res)
    }
    if(model == "individual") return(individual)
    if(model == "pretrain")   return(pretrain)
    if(model == "overall")    return(overall)
}


#' Get the coefficients from a fitted cv.ptLasso model.
#'
#' 
#' @aliases coef.cv.ptLasso 
#' @param object fitted \code{"cv.ptLasso"} object.
#' @param model string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".
#' @param alpha value between 0 and 1, indicating which alpha to use. If \code{NULL}, return coefficients from all models.  Only impacts the results for model = "all" or model = "pretrain".
#' @param \dots other arguments to be passed to the \code{"coef"} function. May be e.g. \code{s = "lambda.min"}.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso}, \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' # Get all model coefficients.
#' names(coef(cvfit))
#'
#' coef(cvfit, model = "overall") # Overall model only
#' length(coef(cvfit, model = "individual")) # List of coefficients for each group model
#' length(coef(cvfit, model = "pretrain", alpha = .5)) # List of coefficients for each group model
#'
#' @method coef cv.ptLasso
#' @export
coef.cv.ptLasso=function(object, model = c("all", "individual", "overall", "pretrain"), alpha = NULL, ...){
    model = match.arg(model)
    
    is.ts = (object$call$use.case == "timeSeries")
    if(is.ts && (model == "overall")) stop("There is no 'overall' model for time series pretraining.")

    if((model == "all") | (model == "individual"))          individual = lapply(object$fitind, function(x) coef(x, ...))
    if(!is.ts && ((model == "all") | (model == "overall"))) overall    = coef(object$fitoverall, ...)

    if((model == "all") | (model == "pretrain")){
        if(is.null(alpha)){
            pretrain = lapply(object$fit, function(model) coef(model, "pretrain", ...))
        } else {
            which.alpha = which(alpha == object$alphalist)
            if(length(which.alpha) == 0) stop("Please choose alpha from fit$alphalist")
            pretrain    = coef(object$fit[[which.alpha]], "pretrain", ...)
        }
    }

    if(is.ts && (model == "all")) return(list(individual = individual,  pretrain = pretrain))
    if(model == "all")            return(list(individual = individual,  pretrain = pretrain, overall = overall))
    if(model == "individual")     return(individual)
    if(model == "pretrain")       return(pretrain)
    if(model == "overall")        return(overall)

}
