#' Get the support for pretrained models
#' 
#' Get the indices of nonzero coefficients from the pretrained models in a fitted ptLasso or cv.ptLasso object, excluding the intercept.
#'
#' @param fit fitted \code{"ptLasso"} or \code{"cv.ptLasso"} object.
#' @param s for glmnet models only: the choice of lambda to use. May be "lambda.min", "lambda.1se" or a numeric value. Default is "lambda.min".
#' @param which for sparsenet models only: the choice of parameters to use. May be "parms.min" or "parms.1se". Default is "parms.min".
#' @param commonOnly whether to return the features that are chosen by all group-specific models (TRUE) or the features that are chosen by any of the group-specific models (FALSE). Default is FALSE.
#' @param includeOverall whether to return the features that are chosen by the overall model and not the group-specific models (TRUE) or the features that are chosen by the overall model or the group-specific models (FALSE). Default is TRUE.
#' @param groups which groups to include when computing the support. Default is to include all groups.
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
get.pretrain.support <- function(fit, s="lambda.min", which="parms.min", commonOnly = FALSE, includeOverall = TRUE, groups = 1:length(fit$fitind)) {
    if(inherits(fit, "cv.ptLasso")) return(lapply(fit$fit, function(m) get.pretrain.or.individual.support(m, s=s, commonOnly=commonOnly, includeOverall=includeOverall, groups = groups, model="pretrain"))) 
    get.pretrain.or.individual.support(fit, s=s, commonOnly=commonOnly, includeOverall=includeOverall, groups = groups, model="pretrain")
}

#' Get the support for individual models
#' 
#' Get the indices of nonzero coefficients from the individual models in a fitted ptLasso or cv.ptLasso object, excluding the intercept.
#'
#' @param fit fitted \code{"ptLasso"} or \code{"cv.ptLasso"} object.
#' @param s for glmnet models only: the choice of lambda to use. May be "lambda.min", "lambda.1se" or a numeric value. Default is "lambda.min".
#' @param which for sparsenet models only: the choice of parameters to use. May be "parms.min" or "parms.1se". Default is "parms.min".
#' @param commonOnly whether to return the features that are chosen by all group-specific models (TRUE) or the features that are chosen by any of the group-specific models (FALSE). Default is FALSE.
#' @param groups which groups to include when computing the support. Default is to include all groups.
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
#'
get.individual.support <- function(fit, s="lambda.min", which="parms.min", commonOnly = FALSE, groups = 1:length(fit$fitind)) {
    if(inherits(fit, "cv.ptLasso")) return(get.pretrain.or.individual.support(fit$fit[[1]], s=s, commonOnly=commonOnly, groups = groups, model="individual")) 
    return(get.pretrain.or.individual.support(fit, s=s, commonOnly=commonOnly, groups = groups, model="individual"))
}

#' Helper function to get support for pretrained or individual models.
#' @noRd
get.pretrain.or.individual.support <- function(fit, s="lambda.min", which="parms.min", model="pretrain", groups = groups, includeOverall=TRUE, commonOnly = FALSE){

    my.coef <- function(model, s, which, ...){
        if(inherits(model, "cv.sparsenet")) return(coef(model, which=which, ...))
        return(coef(model, s=s, ...))
     }
    
    include.these = c()
    if((model == "pretrain") && (includeOverall == TRUE)){
        if(fit$alpha < 1) {
            if(inherits(fit$fitoverall, "cv.sparsenet")) include.these = get.overall.support(fit, which=fit$fitoverall.which)
            if(inherits(fit$fitoverall, "cv.glmnet"))    include.these = get.overall.support(fit, s=fit$fitoverall.lambda)
        }
    }

    model = if(model == "pretrain") {fit$fitpre} else {fit$fitind}
    
    suppre=vector("list",length(groups))
    bhatpre=vector("list",length(groups))
    ix = 1
    for(kk in groups){
        if(fit$call$use.case == "inputGroups"){
            if(fit$call$family == "multinomial"){
                bhatpre[[ix]] = coef(model[[kk]], s=s, exact=FALSE)
                bhatpre[[ix]] = do.call(cbind, bhatpre[[ix]])[-1, ]
                
                suppre[[ix]] = which(apply(bhatpre[[ix]], 1, function(x) sum(x != 0) > 0))
                suppre[[ix]] = sort(unique(c(suppre[[ix]], include.these)))
            } else {
                if(fit$call$family=="cox"){
                    bhatpre[[ix]] = as.numeric(coef(model[[kk]], s=s, exact=F))
                } else {
                    bhatpre[[ix]] = as.numeric(my.coef(model[[kk]], s=s, which=which, exact=F)[-1])
                }
                suppre[[ix]]=sort(unique(c(which(bhatpre[[ix]]!=0), include.these)))
            }
        } else if(fit$call$use.case == "targetGroups"){
            # This should always be a binomial (one vs. rest) model:
            bhatpre[[ix]] = as.numeric(coef(model[[kk]], s=s, exact=F)[-1])
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
#' @param s for glmnet models only: the choice of lambda to use. May be "lambda.min", "lambda.1se" or a numeric value. Default is "lambda.min".
#' @param which for sparsenet models only: the choice of parameters to use. May be "parms.min" or "parms.1se". Default is "parms.min".
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
#'
get.overall.support <- function(fit, s="lambda.min", which="parms.min"){

    if(inherits(fit, "cv.ptLasso")) {
        if("fitoverall.which" %in% names(fit)) {
            if(is.null(which)) which = fit$fitoverall.which
            return(get.overall.support(fit$fit[[1]], which = which))
        }
        if("fitoverall.lambda" %in% names(fit)) {
            if(is.null(s)) s = fit$fitoverall.lambda
            return(get.overall.support(fit$fit[[1]], s = s))
        }
    }

    if(is.null(which) && is.null(s)) stop("s and which cannot both be null.")
    
    if(inherits(fit$fitoverall, "cv.glmnet")) {
        coefs = coef(fit$fitoverall, s=s)
    } else {
        coefs = coef(fit$fitoverall, which=which)
    }
    k = fit$k
    
    # multinomial
    if(is.list(coefs)){
        if(fit$call$use.case == "inputGroups"){
            if(fit$call$family == "cox"){
                return(sort(unique(unlist(lapply(coefs, function(cc) which(cc[-(1:k)] != 0)))))) # first k are group indicators
             } else {
                 return(sort(unique(unlist(lapply(coefs, function(cc) which(cc[-(1:k)] != 0)))))) # first is intercept, next k-1 are indicators
             }
        } else if(fit$call$use.case == "targetGroups") {
            return(sort(unique(unlist(lapply(coefs, function(cc) which(cc[-1] != 0)))))) # no group indicators, only an intercept
        }
    }
    
    # other
    return(which(coefs[-(1:k)] != 0))
}


#' Get the coefficients from a fitted ptLasso model.
#'
#' @aliases coef.ptLasso 
#' @param fit fitted \code{"ptLasso"} object.
#' @param model string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".
#' @param \dots other arguments to be passed to the \code{"coef"} function. For glmnet models, may be e.g. \code{s = "lambda.min"}; for sparsenet models \code{which.parms = "parms.1se"}.
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
#'
coef.ptLasso=function(fit, model = c("all", "individual", "overall", "pretrain"), ...){
    model = match.arg(model)

    if((model == "all") | (model == "individual")) individual = lapply(fit$fitind, function(x) coef(x, ...))
    if((model == "all") | (model == "pretrain"))   pretrain   = lapply(fit$fitpre, function(x) coef(x, ...))
    if((model == "all") | (model == "overall"))    overall    = coef(fit$fitoverall, ...)

    if(model == "all")        return(list( individual = individual,  pretrain = pretrain, overall = overall))
    if(model == "individual") return(individual)
    if(model == "pretrain")   return(pretrain)
    if(model == "overall")    return(overall)

}


#' Get the coefficients from a fitted cv.ptLasso model.
#'
#' 
#' @aliases coef.cv.ptLasso 
#' @param fit fitted \code{"cv.ptLasso"} object.
#' @param model string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".
#' @param alpha value between 0 and 1, indicating which alpha to use. If \code{NULL}, return coefficients from all models.  Only impacts the results for model = "all" or model = "pretrain".
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
coef.cv.ptLasso=function(fit, model = c("all", "individual", "overall", "pretrain"), alpha = NULL, ...){
    model = match.arg(model)

    if((model == "all") | (model == "individual")) individual = lapply(fit$fitind, function(x) coef(x, ...))
    if((model == "all") | (model == "overall"))    overall    = coef(fit$fitoverall, ...)

    if((model == "all") | (model == "pretrain")){
        if(is.null(alpha)){
            pretrain = lapply(fit$fit, function(model) coef(model, "pretrain", ...))
        } else {
            which.alpha = which(alpha == fit$alphalist)
            if(length(which.alpha) == 0) stop("Please choose alpha from fit$alphalist")
            pretrain    = coef(fit$fit[[which.alpha]], "pretrain", ...)
        }
    }

    if(model == "all")        return(list( individual = individual,  pretrain = pretrain, overall = overall))
    if(model == "individual") return(individual)
    if(model == "pretrain")   return(pretrain)
    if(model == "overall")    return(overall)

}
