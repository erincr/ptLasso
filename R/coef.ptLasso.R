
get.pretrain.support <- function(fit, s="lambda.min",  commonOnly = FALSE) get.pretrain.or.individual.support(fit, s=s, commonOnly=commonOnly, model="pretrain")
get.individual.support <- function(fit, s="lambda.min", commonOnly = FALSE) get.pretrain.or.individual.support(fit, s=s, commonOnly=commonOnly, model="individual")

get.pretrain.or.individual.support <- function(fit, s="lambda.min", model="pretrain", commonOnly = FALSE){
    
    include.these = c()
    if(model == "pretrain"){
        if(fit$alpha < 1) include.these = fit$supall
    }

    model = if(model == "pretrain") {fit$fitpre} else {fit$fitind}

    k = length(fit$fitind)
    
    suppre=vector("list",k)
    bhatpre=vector("list",k)
    for(kk in 1:k){
        if(fit$useCase == "inputGroups"){
            if(fit$family == "multinomial"){
                bhatpre[[kk]] = coef(model[[kk]], s=s, exact=FALSE)
                bhatpre[[kk]] = do.call(cbind, bhatpre[[kk]])[-1, ]
                
                suppre[[kk]] = which(apply(bhatpre[[kk]], 1, function(x) sum(x != 0) > 0))
                suppre[[kk]] = sort(unique(c(suppre[[kk]], include.these)))
            } else {
                if(fit$family=="cox"){
                    bhatpre[[kk]] = as.numeric(coef(model[[kk]], s=s, exact=F))
                } else {
                    bhatpre[[kk]] = as.numeric(coef(model[[kk]], s=s, exact=F)[-1])
                }
                suppre[[kk]]=sort(unique(c(which(bhatpre[[kk]]!=0), include.these)))
            }
        } else  if(fit$useCase == "targetGroups"){
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

get.overall.support <- function(fit, s="lambda.min"){

    coefs = coef(fit$fitall, s=s)
    
    # multinomial
    if(is.list(coefs)) return(unique(unlist(lapply(coefs, function(cc) which(cc[-1] != 0)))))
    
    # other
    return(which(coef(fit$fitall, s=s)[-1] != 0))
}


#' Get the coefficients from a fitted ptLasso model.
#'
#' 
#'
#' @aliases coef.ptLasso 
#' @param fit fitted \code{"ptLasso"} object.
#' @param model string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#'
#'
#' @method coef ptLasso
#' @export
#'
#'
coef.ptLasso=function(fit, model = c("all", "individual", "overall", "pretrain")){
    model = match.arg(model)

    if((model == "all") | (model == "individual")) individual = lapply(fit$fitind, coef)
    if((model == "all") | (model == "pretrain"))   pretrain   = lapply(fit$fitpre, coef)
    if((model == "all") | (model == "overall"))    overall    = coef(fit$fitall)

    if(model == "all")        return(list( individual = individual,  pretrain = pretrain, overall = overall))
    if(model == "individual") return(individual)
    if(model == "pretrain")   return(pretrain)
    if(model == "overall")    return(overall)

}


#' Get the coefficients from a fitted cv.ptLasso model.
#'
#' 
#'
#' @aliases coef.cv.ptLasso 
#' @param fit fitted \code{"cv.ptLasso"} object.
#' @param model string indicating which coefficients to retrieve. Must be one of "all", "individual", "overall" or "pretrain".
#' @param alpha value betweein 0 and 1, indicating which alpha to use. If \code{NULL}, return coefficients from all models.  Only impacts the results for model = "all" or model = "pretrain".
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso}, \code{ptLasso}.
#' @keywords models regression classification
#' @examples
#'
#'
#' @method coef cv.ptLasso
#' @export
#'
#'
coef.cv.ptLasso=function(fit, model = c("all", "individual", "overall", "pretrain"), alpha = NULL){
    model = match.arg(model)

    if((model == "all") | (model == "individual")) individual = lapply(fit$fitind, coef)
    if((model == "all") | (model == "overall"))    overall    = coef(fit$fitall)

    if((model == "all") | (model == "pretrain")){
        if(is.null(alpha)){
            pretrain = lapply(fit$fitpre, function(model) lapply(model, coef))
        } else {
            which.alpha = which(alpha == fit$alphalist)
            if(length(which.alpha) == 0) stop("Please choose alpha from fit$alphalist")
            pretrain    = coef(fit$fitpre[[which.alpha]])
        }
    }

    if(model == "all")        return(list( individual = individual,  pretrain = pretrain, overall = overall))
    if(model == "individual") return(individual)
    if(model == "pretrain")   return(pretrain)
    if(model == "overall")    return(overall)

}
