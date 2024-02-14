#' Fit a pretrained lasso model using glmnet.
#'
#' Fits a pretrained lasso model using the glmnet package, for a fixed choice of the pretraining hyperparameter alpha. Additionally fits an "overall" model (using all data) and "individual" models (use each individual group). Can fit input-grouped data with Gaussian, multinomial, binomial or Cox outcomes, and target-grouped data, which necessarily has a multinomial outcome. Many ptLasso arguments are passed directly to glmnet (or sparsenet), and therefore the glmnet (sparsenet) documentation is another good reference for ptLasso.
#'
#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class '"sparseMatrix"' as in package 'Matrix'). Requirement: 'nvars >1'; in other words, 'x' should have 2 or more columns.
#' @param y response variable. Quantitative for 'family="gaussian"'. For 'family="binomial"' should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For 'family="multinomial"', can be a 'nc>=2' level factor, or a matrix with 'nc' columns of counts or proportions. For either '"binomial"' or '"multinomial"', if 'y' is presented as a vector, it will be coerced into a factor. For 'family="cox"', preferably a 'Surv' object from the survival package: see Detail section for more information. For 'family="mgaussian"', 'y' is a matrix of quantitative responses.
#' @param groups A vector of length nobs indicating to which group each observation belongs. For data with k groups, groups should be coded as integers 1 through k. 
#' @param alpha The pretrained lasso hyperparameter, with \eqn{0\le\alpha\le 1}.
#' @param family Either a character string representing one of the built-in families, or else a 'glm()' family object. For more information, see Details section below or the documentation for response type (above).
#' @param type.measure loss to use for cross-validation within each individual, overall, or pretrained lasso model. Currently five options, not all available for all models. The default is 'type.measure="deviance"', which uses squared-error for gaussian models (a.k.a 'type.measure="mse"' there), deviance for logistic and poisson regression, and partial-likelihood for the Cox model. 'type.measure="class"' applies to binomial and multinomial logistic regression only, and gives misclassification error. 'type.measure="auc"' is for two-class logistic regression only, and gives area under the ROC curve. 'type.measure="mse"' or 'type.measure="mae"' (mean absolute error) can be used by all models except the '"cox"'; they measure the deviation from the fitted mean to the response. 'type.measure="C"' is Harrel's concordance measure, only available for 'cox' models.
#' @param use.case The type of grouping observed in the data. Can be one of "inputGroups" or "targetGroups".
#' @param fit.method "glmnet" or "sparsenet". Defaults to "glmnet". If 'fit.method = "glmnet"', then \code{"cv.glmnet"} will be used to train models. If 'fit.method = "sparsenet"', \code{"cv.sparsenet"} will be used. The use of sparsenet is available only when 'family = "gaussian"'.
#' @param overall.lambda For 'fit.method = "glmnet"' only. The choice of lambda to be used by the overall model when defining the offset and penalty factor for pretrained lasso. Defaults to "lambda.1se", but "lambda.min" is another good option. If known in advance, can alternatively supply a numeric value. 
#' @param overall.parms For 'fit.method = "sparsenet"' only. The choice of lambda and gamma to be used by the overall model when defining the offset and penalty factor for pretrained lasso. Can be "parms.1se" or "parms.min". Defaults to "parms.1se".
#' @param fitall An optional cv.glmnet (or cv.sparsenet) object specifying the overall model.
#' @param fitind An optional list of cv.glmnet (or cv.sparsenet) objects specifying the individual models.
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfold} identifying what fold each observation is in. If supplied, \code{nfold} can be missing.
#' @param standardize Should the predictors be standardized before fitting (default is TRUE). If \code{fit.method = "sparsenet"}, standardize must be TRUE.
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit with \code{cv.glmnet}.
#' @param weights observation weights. Default is 1 for each observation.
#' @param penalty.factor Separate penalty factors can be applied to each coefficient. This is a number that multiplies 'lambda' to allow differential shrinkage. Can be 0 for some variables,  which implies no shrinkage, and that variable is always included in the model. Default is 1 for all variables (and implicitly infinity for variables listed in 'exclude'). For more information, see \code{?glmnet}. For pretraining, the user-supplied penalty.factor will be multiplied by the penalty.factor computed by the overall model.
#' @param \dots Additional arguments to be passed to the cv.glmnet (or cv.sparsenet) functions. Notable choices include \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{"cv.glmnet"}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. ptLasso does not support the arguments \code{intercept}, \code{offset}, \code{fit} and \code{check.args}.
#'
#' 
#'
#' @return An object of class \code{"ptLasso"}, which is a list with the ingredients of the fitted models.
#' \item{call}{The call that produced this object.}
#' \item{k}{The number of groups.}
#' \item{alpha}{The value of alpha used for pretraining.}
#' \item{group.levels}{IDs for all of the groups used in training.}
#' \item{fitall}{A fitted \code{cv.glmnet} or \code{cv.sparsenet} object trained using the full data.}
#' \item{fitpre}{A list of fitted (pretrained) \code{cv.glmnet} or \code{cv.sparsenet} objects, one trained with each data group.}
#' \item{fitind}{A list of fitted \code{cv.glmnet} or \code{cv.sparsenet} objects, one trained with each group.}
#' \item{fitall.lambda}{For 'fit.method = "glmnet"'. Lambda used with fitall, to compute the offset for pretraining.}
#' \item{fitall.which}{For 'fit.method = "sparsenet"' only. Gamma and lambda choices used with fitall, to compute the offset for pretraining.}
#' \item{y.mean}{Gaussian outcome only; mean of y for the training data, used for prediction.}
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
#' # plot(fit) to see all of the cv.glmnet models trained
#' predict(fit, xtest, groupstest, ytest=ytest)
#'
#' # Now with sparsenet
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, fit.method = "sparsenet", family = "gaussian", type.measure = "mse")
#' # plot(fit) to see all of the cv.sparsenet models trained
#' predict(fit, xtest, groupstest, ytest=ytest)
#'
#' # Binomial example
#' 
#' out = binomial.example.data()
#' x = out$x; y=out$y; groups = out$group
#'
#' # Test data
#' outtest = binomial.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#' 
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' # plot(fit) to see all of the cv.glmnet models trained
#' predict(fit, xtest, groupstest, ytest=ytest)
#' 
#' @import glmnet sparsenet Matrix
#' @export
#' @seealso \code{\link{glmnet}}, \code{\link{sparsenet}}
#' @references Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1-22.
#' Mazumder, Rahul, Jerome H. Friedman, and Trevor Hastie. "Sparsenet: Coordinate descent with nonconvex penalties." Journal of the American Statistical Association 106.495 (2011): 1125-1138.
#'
#' 
# Note: cv doesn't have to check everything that ptLasso checks
ptLasso=function(x,y,groups,alpha=0.5,family=c("gaussian", "multinomial", "binomial","cox"),
                 type.measure=c("default", "mse", "mae", "auc","deviance","class", "C"),
                 use.case=c("inputGroups","targetGroups"),
                 overall.lambda = "lambda.1se",
                 overall.parms = c("parms.1se", "parms.min"), 
                 fit.method = c("glmnet", "sparsenet"),
                 foldid=NULL,
                 nfolds=10,
                 standardize = TRUE,
                 verbose=FALSE,
                 weights=NULL,
                 penalty.factor = rep(1, nvars),
                 fitall=NULL, fitind=NULL,
                 ...
                 ) {
    this.call = match.call()
    
    family = match.arg(family)
    type.measure = match.arg(type.measure)
    if(type.measure == "default") type.measure = if(family == "gaussian") { "mse" } else { "deviance" }
    use.case = match.arg(use.case, c("inputGroups","targetGroups"), several.ok=FALSE)
    fit.method = match.arg(fit.method, c("glmnet", "sparsenet"), several.ok=FALSE)
    overall.parms = match.arg(overall.parms, c("parms.1se", "parms.min") , several.ok=FALSE)

    if(!(family %in% names(this.call))) this.call$family = family
    if(!(type.measure %in% names(this.call))) this.call$type.measure = type.measure
    if(!(use.case %in% names(this.call))) this.call$use.case = use.case
    
    np=dim(x)
    ##check dims
    if(is.null(np)|(np[2]<=1))stop("x should be a matrix with 2 or more columns")
    nobs=as.integer(np[1])
    nvars=as.integer(np[2])

    k = length(table(groups))
    
    ############################################################################################################
    # Begin error checking:
    ############################################################################################################

    if((fit.method == "sparsenet") & (family != "gaussian")) stop("sparsenet is only available for the Gaussian family.")
    if((fit.method == "sparsenet") & ("exclude" %in% names(list(...)))) stop("sparsenet does not support 'exclude'.")
    if((fit.method == "sparsenet") & (standardize == FALSE)) stop("sparsenet does not support 'standardize = FALSE'.")
    if((fit.method == "sparsenet") & (any(penalty.factor != 1))) stop("sparsenet does not support 'penalty.factor'.")

    if(min(groups) != 1) stop("Groups should be coded from 1 to k.")
    if(length(unique(groups)) < 2) stop("Need to have at least two groups.")
    if(length(unique(groups)) != k) stop(paste0("Expected ", k, " groups, found ", length(unique(groups)), "."))
    if(all(sort(unique(groups)) != (1:k))) stop("Groups should be coded from 1 to k.")

    for(argument in c("fit", "check.args", "offset", "intercept", "standardize.response")){
        if(argument %in% names(list(...))) stop(paste0("ptLasso does not support the argument '", argument, "'."))
    }
    
    if((alpha > 1) | (alpha < 0)) stop("alpha must be between 0 and 1")
    
    # In the future, we want to be able to pass in just the predictions from the overall model.
    # This will be useful for settings where e.g. genentech has released a model (but maybe not as a glmnet object).
    if(!is.null(fitall)){
        if(!("cv.glmnet" %in% class(fitall)) & fit.method == "glmnet" ) stop("fitall must be a cv.glmnet object.")
        if(!("cv.sparsenet" %in% class(fitall)) & fit.method == "sparsenet" ) stop("fitall must be a cv.sparsenet object.")
    }
    if(fit.method == "glmnet" & !is.null(fitind) & !(all(sapply(fitind, function(mm) "cv.glmnet" %in% class(mm))))) stop("fitind must be a list of cv.glmnet objects.")
    if(fit.method == "sparsenet" & !is.null(fitind) & !(all(sapply(fitind, function(mm) "cv.sparsenet" %in% class(mm))))) stop("fitind must be a list of cv.sparsenet objects.")

    if(use.case == "targetGroups" & !(family %in% c("binomial", "multinomial"))){
        stop("Only the multinomial and binomial families are available for target grouped data.")
    }
    
    if(!(type.measure %in% c("class", "deviance")) & family == "multinomial"){
        type.measure = "class"
        message("Only class and deviance are available as type.measure for multinomial models; class used instead.")
    }

    if(type.measure == "auc" & family != "binomial"){
        type.measure = "deviance"
        message("Only the binomial family can use type.measure = auc. Deviance used instead")
    }
    
    if(type.measure == "class" & !(family %in% c("binomial", "multinomial"))){
        type.measure = "deviance"
        message("Only multinomial and binomial families can use type.measure = class. Deviance used instead.")
    }
    ############################################################################################################
    # End error checking
    ############################################################################################################

    p = ncol(x)

    intercept=TRUE   
    if(family=="cox") intercept=FALSE

    method = cv.glmnet
    if(fit.method == "sparsenet") method = my.cv.sparsenet
    
    class.sizes=table(groups)
    
    if(is.null(foldid)){ 
        foldid = rep(1, nrow(x))  
        for(kk in 1:k) foldid[groups == kk] = sample(1:nfolds, class.sizes[kk], replace=TRUE)
        
    }

    if(use.case=="inputGroups"){
        foldid2=vector("list", k)
        for(kk in 1:k) foldid2[[kk]] = sample(rep(1:nfolds,trunc(class.sizes[kk]/nfolds)+1))[1:class.sizes[kk]]
    } else if(use.case=="targetGroups"){
        foldid2=NULL
    }


    ############################################################################################################
    # Fit overall model 
    ############################################################################################################

    group.levels = NULL
    overall.pf = penalty.factor
    if(use.case == "inputGroups") {
        group.levels = sort(unique(groups))
        groups = factor(groups, levels=group.levels)
        onehot.groups = model.matrix(~groups - 1)
        if(family != "cox") onehot.groups = onehot.groups[, 2:k, drop=FALSE]
        overall.pf = c(rep(0, ncol(onehot.groups)), overall.pf)
    }
    
    fitall.is.null = is.null(fitall)
    if(fitall.is.null){
        if(verbose) cat("Fitting overall model",fill=TRUE)

        #strangely, gets upset if you do intercept=FALSE for cox
        if( family!="cox" & use.case == "inputGroups"){
            fitall = method(cbind(onehot.groups, x), y,
                            family=family,
                            foldid=foldid, 
                            intercept=TRUE,
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            ...)
        } else if(family == "cox") {
            fitall = method(cbind(onehot.groups, x), y,
                            family=family,
                            foldid=foldid,  
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            ...)     
        } else if(use.case == "targetGroups") {
            type.multinomial = "grouped"
            if("type.multinomial" %in% names(list(...))) type.multinomial = list(...)$type.multinomial
            fitall = method(x,y,
                            family=family,
                            foldid=foldid,  
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            type.multinomial = type.multinomial,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            ...)     
        }
    }

    if(fit.method == "glmnet"){
        if(overall.lambda == "lambda.min") lamhat = fitall$lambda.min
        if(overall.lambda == "lambda.1se") lamhat = fitall$lambda.1se
        if(is.numeric(overall.lambda)) lamhat = overall.lambda
    }
    if(fit.method == "sparsenet"){
        if(overall.parms == "parms.1se") {which.parms = fitall$which.1se; parms = fitall$parms.1se}
        if(overall.parms == "parms.min") {which.parms = fitall$which.min; parms = fitall$parms.min}
    }

    if(use.case=="inputGroups"){
        if(fit.method == "glmnet"){
            if(family == "multinomial"){
                preval.offset = fitall$fit.preval[, , fitall$lambda == lamhat]
                bhatall=coef(fitall, s=lamhat, exact=FALSE)
                bhatall=do.call(cbind, bhatall)
                bhatall=bhatall[-(1:(k+1)), ]
                supall=which(apply(bhatall, 1, function(x) sum(x != 0) > 0))
                supall=unname(supall)
            } else {
                preval.offset = fitall$fit.preval[, fitall$lambda == lamhat]
                bhatall=as.numeric(coef(fitall, s=lamhat, exact=FALSE))
                if(family!="cox") supall=which(bhatall[-(1:(k+1))]!=0)
                if(family=="cox") supall=which(bhatall[-(1:k)]!=0) 
            }
        } else if(fit.method == "sparsenet") {
            preval.offset = fitall$fit.preval[, which.parms[1], which.parms[2]]
            supall        = which(as.numeric(coef(fitall, which=overall.parms))[-(1:(k+1))] != 0)
        }
    } else if(use.case=="targetGroups"){ # glmnet only!
        preval.offset=vector("list",k)
        bhatall.orig=coef(fitall, s=lamhat, exact=FALSE)
        bhatall=vector("list", k)
        for(kk in 1:k){
            bhatall[[kk]] = as.numeric(bhatall.orig[[kk]])
            preval.offset[[kk]] = fitall$fit.preval[, kk, fitall$lambda == lamhat]
        }
        supall = vector("list",k)
        for(kk in 1:k){ supall[[kk]]=which(bhatall[[kk]][-1]!=0)}
        supall = sort(unique(unlist(supall)))
    }

    ############################################################################################################
    # Fit individual models
    ############################################################################################################
    fitind.is.null = is.null(fitind)

    if(verbose & fitind.is.null) cat("Fitting individual models",fill=TRUE)
    
    if(use.case=="inputGroups"){
        if(fitind.is.null) fitind=vector("list",k) #bhatInd
        
        for(kk in 1:k){
            train.ix = groups == kk
            
            # individual model 
            if(fitind.is.null){
                if(family!="cox") { 
                    fitind[[kk]]=method(x[train.ix,], y[train.ix],
                                        family=family,
                                        type.measure=type.measure,
                                        foldid=foldid2[[kk]],
                                        intercept=intercept,
                                        penalty.factor=penalty.factor,
                                        weights=weights[train.ix],
                                        keep=TRUE,
                                        standardize=standardize,
                                        ...)
                } else if(family=="cox") {
                    fitind[[kk]]=method(x[train.ix,], y[train.ix, ],
                                        family=family,
                                        type.measure=type.measure,
                                        foldid=foldid2[[kk]],
                                        weights=weights[train.ix],
                                        keep=TRUE,
                                        standardize=standardize,
                                        ...)
                }
            }
        }
    }
   
    if(use.case=="targetGroups"){
        for(kk in 1:k){
            if(fitind.is.null){
                yy = rep(0, nrow(x))
                yy[y == kk]=1
                
                fitind[[kk]] = method(x,yy,
                                      family="binomial",
                                      foldid=foldid,
                                      type.measure=type.measure,
                                      penalty.factor=penalty.factor,
                                      keep=TRUE,
                                      weights=weights,
                                      standardize=standardize,
                                      ...)
           }
        }
    }
            
    ####################################################################################
    # Now, fit pretrained models 
    ####################################################################################
    if(verbose) cat("Fitting pretrained lasso models",fill=TRUE)
    
    fitpre=vector("list",k)

    if(alpha == 1){
        fitpre = fitind
    } else {
         if(use.case=="inputGroups"){
             for(kk in 1:k){ 
                 train.ix = groups == kk

                 if(family == "multinomial"){
                     offset = (1-alpha) * preval.offset[train.ix, ]
                 } else {
                     offset = (1-alpha) * preval.offset[train.ix]
                 }
                 
                 fac = rep(1/alpha, p)
                 fac[supall] = 1
                 pf = penalty.factor * fac 

                 if((alpha == 0) & (length(supall) == 0)) {
                     almost.zero = 1e-9
                     fac = rep(1/almost.zero, p)
                     fac[supall] = 1
                     pf = penalty.factor * fac
                 }

                 if(family!="cox") fitpre[[kk]] = method(x[train.ix,],
                                                         y[train.ix],
                                                         family=family, 
                                                         offset=offset,
                                                         intercept=intercept,
                                                         type.measure=type.measure,
                                                         foldid=foldid2[[kk]],
                                                         penalty.factor=pf,
                                                         weights=weights[train.ix],
                                                         keep=TRUE,
                                                         standardize=standardize,
                                                         ...)
                 if(family=="cox") fitpre[[kk]] = method(x[train.ix,],
                                                         y[train.ix,],
                                                         family=family, 
                                                         offset=offset,
                                                         type.measure=type.measure,
                                                         penalty.factor=pf,
                                                         foldid=foldid2[[kk]],
                                                         weights=weights[train.ix],
                                                         keep=TRUE,
                                                         standardize=standardize,
                                                         ...)
             }
         } else if(use.case=="targetGroups"){
             for(kk in 1:k){ 
                 
                 myoffset = (1-alpha) * preval.offset[[kk]]
                 
                 pf = rep(1/alpha, p)
                 pf[supall[[kk]]] = 1
                 pf = pf * penalty.factor

                 yy=rep(0,nrow(x)) 
                 yy[y==kk]=1  

                 fitpre[[kk]] = method(x,yy,
                                       family="binomial", 
                                       offset=myoffset,
                                       penalty.factor=pf,
                                       foldid=foldid,
                                       type.measure=type.measure,
                                       weights=weights,
                                       keep=TRUE,
                                       standardize=standardize,
                                       ...)
                 
                 
                 
             }
         }
     }

   
    out=enlist(
               # Info about the initial call:
               call=this.call,
               k, alpha, group.levels,
               
               # Fitted models
               fitall, fitind, fitpre
    )
    if(fit.method == "glmnet") out$fitall.lambda = lamhat
    if(fit.method == "sparsenet") out$fitall.which = overall.parms
    class(out)="ptLasso"
    return(out)

}

my.cv.sparsenet <- function(x, y, ...){
    params         = list(...)
    remove.params  = c("family", "intercept", "offset", "standardize", "penalty.factor")
    out            = list()
    
    # Remove offset - gaussian only
    params[["x"]] = x
    if("offset" %in% names(params)){
        params[["y"]] = y - params[["offset"]]
    } else {
        params[["y"]] = y
    }

    if( ("weights" %in%  names(params)) & is.null(params[["weights"]]) ) params[["weights"]] = rep(1, nrow(x))

    if( any(params[["penalty.factor"]] == 0) ){
        icepts = lm(y ~ x[, params[["penalty.factor"]] == 0])
        icepts = coef(icepts)[-1]
        params[["y"]] = params[["y"]] - colSums( icepts * t(x[, params[["penalty.factor"]] == 0]) )

        params[["x"]] =  params[["x"]][, params[["penalty.factor"]] != 0]
    }

    model = do.call(cv.sparsenet, params[!(names(params) %in% remove.params)])

    if( any(params[["penalty.factor"]] == 0) ){
        nc = length(model$sparsenet.fit$lambda)
        icept.rows = t(sapply(icepts, function(i) rep(i, nc)))
        for(ix in 1:length(model$sparsenet.fit$gamma)){
            model$sparsenet.fit$coefficients[[ix]]$beta = rbind(icept.rows, model$sparsenet.fit$coefficients[[ix]]$beta)
        }
    }

    if("offset" %in% names(params)){
        model$fit.preval = model$fit.preval + params[["offset"]]
    }

    
    return(model)
}
