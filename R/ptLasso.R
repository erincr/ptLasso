#' Fit a pretrained lasso model using glmnet.
#'
#' Fits a pretrained lasso model using the glmnet package, for a fixed choice of the pretraining hyperparameter alpha. Additionally fits an "overall" model (using all data) and "individual" models (use each individual group). Can fit input-grouped data with Gaussian, multinomial, binomial or Cox outcomes, and target-grouped data, which necessarily has a multinomial outcome. Many ptLasso arguments are passed directly to glmnet, and therefore the glmnet documentation is another good reference for ptLasso.
#'
#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class '"sparseMatrix"' as in package 'Matrix'). Requirement: 'nvars >1'; in other words, 'x' should have 2 or more columns.
#' @param y response variable. Quantitative for 'family="gaussian"'. For 'family="binomial"' should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For 'family="multinomial"', can be a 'nc>=2' level factor, or a matrix with 'nc' columns of counts or proportions. For either '"binomial"' or '"multinomial"', if 'y' is presented as a vector, it will be coerced into a factor. For 'family="cox"', preferably a 'Surv' object from the survival package: see Detail section for more information. For 'family="mgaussian"', 'y' is a matrix of quantitative responses.
#' @param groups A vector of length nobs indicating to which group each observation belongs. For data with k groups, groups should be coded as integers 1 through k. 
#' @param alpha The pretrained lasso hyperparameter, with \eqn{0\le\alpha\le 1}. The range of alpha is from 0 (which fits the overall model with fine tuning) to 1 (the individual models). The default value is 0.5, chosen mostly at random. To choose the appropriate value for your data, please either run \code{ptLasso} with a few choices of alpha and evaluate with a validation set, or use cv.ptLasso, which recommends a value of alpha using cross validation.
#' @param family Either a character string representing one of the built-in families, or else a 'glm()' family object. For more information, see Details section below or the documentation for response type (above).
#' @param type.measure loss to use for cross-validation within each individual, overall, or pretrained lasso model. Currently five options, not all available for all models. The default is 'type.measure="deviance"', which uses squared-error for gaussian models (a.k.a 'type.measure="mse"' there), deviance for logistic and poisson regression, and partial-likelihood for the Cox model. 'type.measure="class"' applies to binomial and multinomial logistic regression only, and gives misclassification error. 'type.measure="auc"' is for two-class logistic regression only, and gives area under the ROC curve. 'type.measure="mse"' or 'type.measure="mae"' (mean absolute error) can be used by all models except the '"cox"'; they measure the deviation from the fitted mean to the response. 'type.measure="C"' is Harrel's concordance measure, only available for 'cox' models.
#' @param use.case The type of grouping observed in the data. Can be one of "inputGroups" or "targetGroups".
#' @param overall.lambda The choice of lambda to be used by the overall model to define the offset and penalty factor for pretrained lasso. Defaults to "lambda.1se", but "lambda.min" is another good option. If known in advance, can alternatively supply a numeric value. This choice of lambda will be used to compute the offset and penalty factor (1) during model training and (2) during prediction. In the predict function, another lambda must be specified for the individual models, the second stage of pretraining and the overall model.
#' @param overall.gamma For use only when the option \code{relax = TRUE} is specified. The choice of gamma to be used by the overall model to define the offset and penalty factor for pretrained lasso. Defaults to "gamma.1se", but "gamma.min" is also a good option. This choice of gamma will be used to compute the offset and penalty factor (1) during model training and (2) during prediction. In the predict function, another gamma must be specified for the individual models, the second stage of pretraining and the overall model.
#' @param fitoverall An optional cv.glmnet object specifying the overall model. This should have been trained on the full training data, with the argumnet keep = TRUE.
#' @param fitind An optional list of cv.glmnet objects specifying the individual models. 
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfold} identifying what fold each observation is in. If supplied, \code{nfold} can be missing.
#' @param standardize Should the predictors be standardized before fitting (default is TRUE). 
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit with \code{cv.glmnet}.
#' @param weights observation weights. Default is 1 for each observation.
#' @param penalty.factor Separate penalty factors can be applied to each coefficient. This is a number that multiplies 'lambda' to allow differential shrinkage. Can be 0 for some variables,  which implies no shrinkage, and that variable is always included in the model. Default is 1 for all variables (and implicitly infinity for variables listed in 'exclude'). For more information, see \code{?glmnet}. For pretraining, the user-supplied penalty.factor will be multiplied by the penalty.factor computed by the overall model.
#' @param en.alpha The elasticnet mixing parameter, with 0 <= en.alpha <= 1. The penalty is defined as (1-alpha)/2||beta||_2^2+alpha||beta||_1. 'alpha=1' is the lasso penalty, and 'alpha=0' the ridge penalty. Default is `en.alpha = 1` (lasso).
#' @param group.intercepts For 'use.case = "inputGroups"' only. If `TRUE`, fit the overall model with a separate intercept for each group. If `FALSE`, ignore the grouping and fit one overall intercept. Default is `TRUE`.
#' @param \dots Additional arguments to be passed to the cv.glmnet functions. Notable choices include \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{"cv.glmnet"}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. ptLasso does not support the arguments \code{intercept}, \code{offset}, \code{fit} and \code{check.args}.
#'
#' 
#'
#' @return An object of class \code{"ptLasso"}, which is a list with the ingredients of the fitted models.
#' \item{call}{The call that produced this object.}
#' \item{k}{The number of groups.}
#' \item{alpha}{The value of alpha used for pretraining.}
#' \item{group.levels}{IDs for all of the groups used in training.}
#' \item{fitoverall}{A fitted \code{cv.glmnet} object trained using the full data.}
#' \item{fitpre}{A list of fitted (pretrained) \code{cv.glmnet} objects, one trained with each data group.}
#' \item{fitind}{A list of fitted \code{cv.glmnet} objects, one trained with each group.}
#' \item{fitoverall.lambda}{Lambda used with fitoverall, to compute the offset for pretraining.}
#' 
#' @examples
#' # Getting started. First, we simulate data: we need covariates x, response y and group IDs.
#' set.seed(1234)
#' x = matrix(rnorm(1000*20), 1000, 20)
#' y = rnorm(1000)
#' groups = sort(rep(1:5, 200))
#'
#' xtest = matrix(rnorm(1000*20), 1000, 20)
#' ytest = rnorm(1000)
#' groupstest = sort(rep(1:5, 200))
#'
#' # Now, we can fit a ptLasso model:
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' # plot(fit) # to see all of the cv.glmnet models trained
#' predict(fit, xtest, groupstest) # to predict on new data
#' predict(fit, xtest, groupstest, ytest=ytest) # if ytest is included, we also measure performance
#'
#' # When we trained our model, we used "lambda.1se" in the first stage of pretraining by default.
#' # This is a necessary choice to make during model training; we need to select the model
#' # we want to use to define the offset and penalty factor for the second stage of pretraining.
#' # We could instead have used "lambda.min":
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse",
#'               overall.lambda = "lambda.min")
#'
#' # We can use the 'relax' option to fit relaxed lasso models:
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5,
#'               family = "gaussian", type.measure = "mse",
#'               relax = TRUE)
#'
#' # As we did for lambda, we may want to specify the choice of gamma for stage one
#' # of pretraining. (The default is "gamma.1se".)
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5,
#'               family = "gaussian", type.measure = "mse",
#'               relax = TRUE, overall.gamma = "gamma.min")
#'
#' # In practice, we may want to try many values of alpha.
#' # alpha may range from 0 (the overall model with fine tuning) to 1 (the individual models).
#' # To choose alpha, you may either (1) run ptLasso with different values of alpha
#' # and measure performance with a validation set, or (2) use cv.ptLasso.
#'
#' 
#' # Now, we are ready to simulate slightly more realistic data.
#' # This continuous outcome example has k = 5 groups, where each group has 200 observations.
#' # There are scommon = 10 features shared across all groups, and
#' # sindiv = 10 features unique to each group.
#' # n = 1000 and p = 120 (60 informative features and 60 noise features).
#' # The coefficients of the common features differ across groups (beta.common).
#' # In group 1, these coefficients are rep(1, 10); in group 2 they are rep(2, 10), etc.
#' # Each group has 10 unique features, the coefficients of which are all 3 (beta.indiv).
#' # The intercept in all groups is 0.
#' # The variable sigma = 20 indicates that we add noise to y according to 20 * rnorm(n). 
#' set.seed(1234)
#' k=5
#' class.sizes=rep(200, k)
#' scommon=10; sindiv=rep(10, k)
#' n=sum(class.sizes); p=2*(sum(sindiv) + scommon)
#' beta.common=3*(1:k); beta.indiv=rep(3, k)
#' intercepts=rep(0, k)
#' sigma=20
#' out = gaussian.example.data(k=k, class.sizes=class.sizes,
#'                             scommon=scommon, sindiv=sindiv,
#'                             n=n, p=p,
#'                             beta.common=beta.common, beta.indiv=beta.indiv,
#'                             intercepts=intercepts, sigma=20)
#' x = out$x; y=out$y; groups = out$group
#'
#' outtest = gaussian.example.data(k=k, class.sizes=class.sizes,
#'                                 scommon=scommon, sindiv=sindiv,
#'                                 n=n, p=p,
#'                                 beta.common=beta.common, beta.indiv=beta.indiv,
#'                                 intercepts=intercepts, sigma=20)
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#' 
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' # plot(fit) to see all of the cv.glmnet models trained
#' predict(fit, xtest, groupstest, ytest=ytest)
#'
#' # Now, we repeat with a binomial outcome.
#' # This example has k = 3 groups, where each group has 100 observations.
#' # There are scommon = 5 features shared across all groups, and
#' # sindiv = 5 features unique to each group.
#' # n = 300 and p = 40 (20 informative features and 20 noise features).
#' # The coefficients of the common features differ across groups (beta.common),
#' # as do the coefficients specific to each group (beta.indiv).
#' set.seed(1234)
#' k=3
#' class.sizes=rep(100, k)
#' scommon=5; sindiv=rep(5, k)
#' n=sum(class.sizes); p=2*(sum(sindiv) + scommon)
#' beta.common=list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3))
#' beta.indiv = lapply(1:k, function(i)  0.9 * beta.common[[i]])
#' 
#' out = binomial.example.data(k=k, class.sizes=class.sizes,
#'                             scommon=scommon, sindiv=sindiv,
#'                             n=n, p=p,
#'                             beta.common=beta.common, beta.indiv=beta.indiv)
#' x = out$x; y=out$y; groups = out$group
#'
#' outtest = binomial.example.data(k=k, class.sizes=class.sizes,
#'                                 scommon=scommon, sindiv=sindiv,
#'                                 n=n, p=p,
#'                                 beta.common=beta.common, beta.indiv=beta.indiv)
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#' 
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' # plot(fit) to see all of the cv.glmnet models trained
#' predict(fit, xtest, groupstest, ytest=ytest)
#'
#' \dontrun{
#' ### Model fitting with parallel = TRUE
#' require(doMC)
#' registerDoMC(cores = 4)
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse", parallel=TRUE)
#' }
#' 
#' 
#' @import glmnet Matrix
#' @export
#' @seealso \code{\link{glmnet}}
#' @references Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1-22.
#'
#' 
ptLasso=function(x,y,groups,alpha=0.5,family=c("gaussian", "multinomial", "binomial","cox"),
                 type.measure=c("default", "mse", "mae", "auc","deviance","class", "C"),
                 use.case=c("inputGroups","targetGroups"),
                 overall.lambda = "lambda.1se",
                 overall.gamma = "gamma.1se",
                 foldid=NULL,
                 nfolds=10,
                 standardize = TRUE,
                 verbose=FALSE,
                 weights=NULL,
                 penalty.factor = rep(1, nvars),
                 fitoverall=NULL, fitind=NULL,
                 en.alpha = 1,
                 group.intercepts = TRUE,
                 ...
                 ) {
    this.call = match.call()
    
    family = match.arg(family)
    type.measure = match.arg(type.measure)
    if(type.measure == "default") type.measure = if(family == "gaussian") { "mse" } else { "deviance" }
    use.case = match.arg(use.case, c("inputGroups","targetGroups"), several.ok=FALSE)

    if(!(family %in% names(this.call))) this.call$family = family
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

    if( (use.case == "targetGroups") && (family != "multinomial") ) stop("use.case = 'targetGroups' is only possible for family = 'multinomial'.")
    
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
    if(!is.null(fitoverall)){
        if(!("cv.glmnet" %in% class(fitoverall))) stop("fitoverall must be a cv.glmnet object.")
        if(!("fit.preval" %in% names(fitoverall))) stop("fitoverall must have fit.preval defined (fitted with the argument keep = TRUE).")
        if(nrow(get.preval(fitoverall, gamma = overall.gamma)) != nrow(x)) stop("fitoverall must have been trained using the same training data passed to ptLasso.")
    }
    if(!is.null(fitind)){
        if(length(fitind) != k) stop("Some of the individual models are missing: need one model trained for each group.")
        if(!(all(sapply(fitind, function(mm) "cv.glmnet" %in% class(mm))))) stop("fitind must be a list of cv.glmnet objects.")
        if(!all(sapply(fitind, function(mm) "fit.preval" %in% names(mm)))) stop("Individual models must have fit.preval defined (fitted with the argument keep = TRUE).")
        if(use.case == "inputGroups"){
            if(!all(sapply(fitind, function(mm) nrow(get.preval(mm, gamma = overall.gamma))) == table(groups))) stop("Individual models must have been trained using the same training data passed to ptLasso.")
        } else {
            if(!all(sapply(fitind, function(mm) nrow(get.preval(mm, gamma = overall.gamma))) == nrow(x))) stop("Individual models must have been trained using the same training data passed to ptLasso.")                                                                                                                                                                              }
    }
    

    type.measure = cvtype(type.measure=type.measure,family=family)
    this.call$type.measure = type.measure

    this.call$group.intercepts = group.intercepts
    
    ############################################################################################################
    # End error checking
    ############################################################################################################

    p = ncol(x)

    intercept=TRUE   
    if(family=="cox") intercept=FALSE
    
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

    fitind.is.null = is.null(fitind)
    fitoverall.is.null = is.null(fitoverall)
    ############################################################################################################
    # Fit overall model 
    ############################################################################################################

    group.levels = NULL
    overall.pf = penalty.factor
    onehot.groups = NULL
    if(use.case == "inputGroups" && group.intercepts == TRUE) {
        group.levels = sort(unique(groups))
        groups = factor(groups, levels=group.levels)
        onehot.groups = model.matrix(~groups - 1)
        if(family != "cox") onehot.groups = onehot.groups[, 2:k, drop=FALSE]
        overall.pf = c(rep(0, ncol(onehot.groups)), overall.pf)
    }

    if(fitoverall.is.null){
        if(verbose) cat("Fitting overall model",fill=TRUE)

        #strangely, gets upset if you do intercept=FALSE for cox
        if( family!="cox" & use.case == "inputGroups"){
            fitoverall = cv.glmnet(cbind(onehot.groups, x), y,
                            family=family,
                            foldid=foldid, 
                            intercept=TRUE,
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            alpha=en.alpha,
                            ...)
        } else if(family == "cox") {
            fitoverall = cv.glmnet(cbind(onehot.groups, x), y,
                            family=family,
                            foldid=foldid,  
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            alpha=en.alpha,
                            ...)     
        } else if(use.case == "targetGroups") {
            type.multinomial = "grouped"
            if("type.multinomial" %in% names(list(...))) type.multinomial = list(...)$type.multinomial
            fitoverall = cv.glmnet(x,y,
                            family=family,
                            foldid=foldid,  
                            type.measure=type.measure,
                            penalty.factor=overall.pf,
                            type.multinomial = type.multinomial,
                            keep=TRUE,
                            weights=weights,
                            standardize=standardize,
                            alpha=en.alpha,
                            ...)     
        }
    }

    if(overall.lambda == "lambda.min") lamhat = fitoverall$lambda.min
    if(overall.lambda == "lambda.1se") lamhat = fitoverall$lambda.1se
    if(is.numeric(overall.lambda)) lamhat = overall.lambda

    if(use.case=="inputGroups"){
        if(family == "multinomial"){
            preval.offset = get.preval(fitoverall, gamma = overall.gamma)[, , fitoverall$lambda == lamhat]
            bhatall=coef(fitoverall, s=lamhat, exact=FALSE)
            bhatall=do.call(cbind, bhatall)
            bhatall=bhatall[-(1:(k+1)), ]
            supall=which(apply(bhatall, 1, function(x) sum(x != 0) > 0))
            supall=unname(supall)
        } else {
            preval.offset = get.preval(fitoverall, gamma = overall.gamma)[, fitoverall$lambda == lamhat]
            bhatall=as.numeric(coef(fitoverall, s=lamhat, exact=FALSE))
            if(family!="cox") supall=which(bhatall[-(1:(k+1))]!=0)
            if(family=="cox") supall=which(bhatall[-(1:k)]!=0)
        } 
    } else if(use.case=="targetGroups"){
        preval.offset=vector("list",k)
        bhatall.orig=coef(fitoverall, s=lamhat, exact=FALSE)
        bhatall=vector("list", k)
        for(kk in 1:k){
            bhatall[[kk]] = as.numeric(bhatall.orig[[kk]])
            preval.offset[[kk]] = get.preval(fitoverall, gamma = overall.gamma)[, kk, fitoverall$lambda == lamhat]
        }
        supall = vector("list",k)
        for(kk in 1:k){ supall[[kk]]=which(bhatall[[kk]][-1]!=0)}
        supall = sort(unique(unlist(supall)))
    }

    ############################################################################################################
    # Fit individual models
    ############################################################################################################

    if(verbose & fitind.is.null) cat("Fitting individual models",fill=TRUE)
    
    if(fitind.is.null && (use.case=="inputGroups")){
        if(fitind.is.null) fitind=vector("list",k) #bhatInd
        
        for(kk in 1:k){
            train.ix = groups == kk
            
            # individual model 
            if(verbose) cat("\tFitting individual model", kk, "/", k, fill=TRUE)
            if(family!="cox") { 
                    fitind[[kk]] = cv.glmnet(x[train.ix,], y[train.ix],
                                        family=family,
                                        type.measure=type.measure,
                                        foldid=foldid2[[kk]],
                                        intercept=intercept,
                                        penalty.factor=penalty.factor,
                                        weights=weights[train.ix],
                                        keep=TRUE,
                                        standardize=standardize,
                                        alpha=en.alpha,
                                        ...)
             } else if(family=="cox") {
                    fitind[[kk]] = cv.glmnet(x[train.ix,], y[train.ix, ],
                                        family=family,
                                        type.measure=type.measure,
                                        foldid=foldid2[[kk]],
                                        weights=weights[train.ix],
                                        keep=TRUE,
                                        standardize=standardize,
                                        alpha=en.alpha,
                                        ...)
            }
        }
    }
   
    if(use.case=="targetGroups"){
        for(kk in 1:k){
            if(fitind.is.null){
                yy = rep(0, nrow(x))
                yy[y == kk]=1
                
                fitind[[kk]] = cv.glmnet(x,yy,
                                      family="binomial",
                                      foldid=foldid,
                                      type.measure=type.measure,
                                      penalty.factor=penalty.factor,
                                      keep=TRUE,
                                      weights=weights,
                                      standardize=standardize,
                                      alpha=en.alpha,
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
                 if(verbose) cat("\tFitting pretrained model", kk, "/", k, fill=TRUE)
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

                 if(family!="cox") fitpre[[kk]] = cv.glmnet(x[train.ix,],
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
                                                         alpha=en.alpha,
                                                         ...)
                 if(family=="cox") fitpre[[kk]] = cv.glmnet(x[train.ix,],
                                                         y[train.ix,],
                                                         family=family, 
                                                         offset=offset,
                                                         type.measure=type.measure,
                                                         penalty.factor=pf,
                                                         foldid=foldid2[[kk]],
                                                         weights=weights[train.ix],
                                                         keep=TRUE,
                                                         standardize=standardize,
                                                         alpha=en.alpha,
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

                 fitpre[[kk]] = cv.glmnet(x,yy,
                                       family="binomial", 
                                       offset=myoffset,
                                       penalty.factor=pf,
                                       foldid=foldid,
                                       type.measure=type.measure,
                                       weights=weights,
                                       keep=TRUE,
                                       standardize=standardize,
                                       alpha=en.alpha,
                                       ...)
                 
                 
                 
             }
         }
     }

   
    out=enlist(
        # Info about the initial call:
        call=this.call,
        k, alpha, group.levels,
               
        # Fitted models
        fitoverall, fitind, fitpre,
        fitoverall.lambda = lamhat
    )
    if(("relax" %in% names(list(...))) && list(...)$relax == TRUE) out$fitoverall.gamma = overall.gamma
    class(out)="ptLasso"
    return(out)

}

#' Error checking for type.measure and family - modified from glmnet cvtype.R
#' @noRd
cvtype <- function(type.measure="mse",family="gaussian"){
    type.measures = c("mse","deviance", "class", "auc", "mae","C")
    devname=switch(family,
                   "gaussian"="Mean-squared Error",
                   "binomial"="Binomial Deviance",
                   "cox"="Partial Likelihood Deviance",
                   "multinomial"="Multinomial Deviance"
                   )
    typenames = c(deviance = devname, mse = "Mean-Squared Error",
    mae = "Mean Absolute Error",auc = "AUC", class = "Misclassification Error",C="C-index")
    subclass.ch=switch(family,
                   "gaussian"=c(1,2,5),
                   "binomial"=c(2,3,4),
                   "cox"=c(2,6),
                   "multinomial"=c(2,3)
                   )
   subclass.type=type.measures[subclass.ch]
   if(type.measure=="default")type.measure=subclass.type[1]
    model.name=switch(family,
                   "gaussian"="Gaussian",
                   "binomial"="Binomial",
                   "cox"="Cox",
                   "multinomial"="Multinomial"
                   )
    if(!match(type.measure,subclass.type,FALSE)){
        type.measure=subclass.type[1]
        warning(paste("Only ",paste(subclass.type,collapse=", ")," available as type.measure for ",model.name," models; ", type.measure," used instead",sep=""),call.=FALSE)
    }
    type.measure
}

#' Error checking for type.measure and family - modified from glmnet cvtype.R
#' @noRd
get.preval <- function(fit, gamma = "gamma.1se"){
    if("relaxed" %in% names(fit)){
        if(!(gamma %in% c("gamma.1se", "gamma.min"))) stop("gamma must be 'gamma.1se' or 'gamma.min'.")
        if(gamma == "gamma.1se") ix = which(fit$relaxed$gamma == fit$relaxed$gamma.1se)
        if(gamma == "gamma.min") ix = which(fit$relaxed$gamma == fit$relaxed$gamma.min)
        if(is.numeric(gamma)) ix = which(abs(fit$relaxed$gamma - gamma) < 1e-5)
        if(length(ix) < 1) stop("gamma must be in fit$relaxed$gamma.")
        if(length(ix) > 1) ix = ix[1]
        return(fit$fit.preval[[ix]])
    } else {
        return(fit$fit.preval)
    }
            
}
    
