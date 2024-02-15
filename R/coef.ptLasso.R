#' Get the support for pretrained models
#' @noRd
get.pretrain.support <- function(fit, s="lambda.min", which="parms.min", commonOnly = FALSE) {
    if(inherits(fit, "cv.ptLasso")) return(lapply(fit$fit, function(model) get.pretrain.or.individual.support(model, s=s, commonOnly=commonOnly, model="pretrain"))) 
    get.pretrain.or.individual.support(fit, s=s, commonOnly=commonOnly, model="pretrain")
}
#' Get the support for individual models
#' @noRd
get.individual.support <- function(fit, s="lambda.min", which="parms.min", commonOnly = FALSE) {
    if(inherits(fit, "cv.ptLasso")) return(get.pretrain.or.individual.support(fit$fit[[1]], s=s, commonOnly=commonOnly, model="individual")) 
    return(get.pretrain.or.individual.support(fit, s=s, commonOnly=commonOnly, model="individual"))
}
#' Helper function to get support for pretrained or individual models.
#' @noRd
get.pretrain.or.individual.support <- function(fit, s="lambda.min", which="parms.min", model="pretrain", commonOnly = FALSE){

    my.coef <- function(model, s, which, ...){
        if(inherits(model, "cv.sparsenet")) return(coef(model, which=which, ...))
        return(coef(model, s=s, ...))
     }
    
    include.these = c()
    if(model == "pretrain"){
        if(fit$alpha < 1) {
            if(inherits(model, "cv.sparsenet")) include.these = get.overall.support(fit, s=fit$fitall.lambda)
            if(inherits(model, "cv.glmnet")) include.these = get.overall.support(fit, s=fit$fitall.lambda)
        }
    }

    model = if(model == "pretrain") {fit$fitpre} else {fit$fitind}

    k = length(fit$fitind)
    
    suppre=vector("list",k)
    bhatpre=vector("list",k)
    for(kk in 1:k){
        if(fit$call$use.case == "inputGroups"){
            if(fit$call$family == "multinomial"){
                bhatpre[[kk]] = coef(model[[kk]], s=s, exact=FALSE)
                bhatpre[[kk]] = do.call(cbind, bhatpre[[kk]])[-1, ]
                
                suppre[[kk]] = which(apply(bhatpre[[kk]], 1, function(x) sum(x != 0) > 0))
                suppre[[kk]] = sort(unique(c(suppre[[kk]], include.these)))
            } else {
                if(fit$call$family=="cox"){
                    bhatpre[[kk]] = as.numeric(coef(model[[kk]], s=s, exact=F))
                } else {
                    bhatpre[[kk]] = as.numeric(my.coef(model[[kk]], s=s, which=which, exact=F)[-1])
                }
                suppre[[kk]]=sort(unique(c(which(bhatpre[[kk]]!=0), include.these)))
            }
        } else  if(fit$call$use.case == "targetGroups"){
            # This should always be a binomial (one vs. rest) model:
            bhatpre[[kk]] = as.numeric(coef(model[[kk]], s=s, exact=F)[-1])
            suppre[[kk]] = sort(unique(c(which(bhatpre[[kk]]!=0), include.these)))
        }
    }

    all.selected = sort(unique(unlist(suppre)))
    if(!commonOnly) return(all.selected)

    counts = sapply(all.selected, function(coeff) sum(sapply(suppre, function(supp) coeff %in% supp)))
    return(all.selected[counts > k/2])
}

#' Get the support for the overall model
#' @noRd
get.overall.support <- function(fit, s=NULL, which=NULL){

    if(inherits(fit, "cv.ptLasso")) {
        if("fitall.which" %in% names(fit)) {
            if(is.null(which)) which = fit$fitall.which
            return(get.overall.support(fit$fit[[1]], which = which))
        }
        if("fitall.lambda" %in% names(fit)) {
            if(is.null(s)) s = fit$fitall.lambda
            return(get.overall.support(fit$fit[[1]], s = s))
        }
    }

    if(is.null(which) & is.null(s)) stop("s and which cannot both be null.")
    
    if(inherits(fit$fitall, "cv.glmnet")) {
        coefs = coef(fit$fitall, s=s)
    } else {
        coefs = coef(fit$fitall, which=which)
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
    if((model == "all") | (model == "overall"))    overall    = coef(fit$fitall, ...)

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
coef.cv.ptLasso=function(fit, model = c("all", "individual", "overall", "pretrain"), alpha = NULL){
    model = match.arg(model)

    if((model == "all") | (model == "individual")) individual = lapply(fit$fitind, coef)
    if((model == "all") | (model == "overall"))    overall    = coef(fit$fitall)

    if((model == "all") | (model == "pretrain")){
        if(is.null(alpha)){
            pretrain = lapply(fit$fit, function(model) coef(model, "pretrain"))
        } else {
            which.alpha = which(alpha == fit$alphalist)
            if(length(which.alpha) == 0) stop("Please choose alpha from fit$alphalist")
            pretrain    = coef(fit$fit[[which.alpha]], "pretrain")
        }
    }

    if(model == "all")        return(list( individual = individual,  pretrain = pretrain, overall = overall))
    if(model == "individual") return(individual)
    if(model == "pretrain")   return(pretrain)
    if(model == "overall")    return(overall)

}
