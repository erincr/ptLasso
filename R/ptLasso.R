#' Fit a pretrained lasso model using glmnet.
#'
#' Fits a pretrained lasso model using the glmnet package, for a fixed choice of the pretraining hyperparameter alpha. Additionally fits an "overall" model (using all data) and "individual" models (use each individual group). Can fit input-grouped data with Gaussian, multinomial, binomial or Cox outcomes, and target-grouped data, which necessarily has a multinomial outcome. Many ptLasso arguments are passed directly to glmnet, and therefore the glmnet documentation is another good reference for ptLasso.
#'
#' Importantly, ptLasso performs cross validation to select the lasso hyperparameter lambda via cv.glmnet. To choose the pretrained hyperparameter alpha, please use cv.ptLasso.
#'
#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class '"sparseMatrix"' as in package 'Matrix'). Requirement: 'nvars >1'; in other words, 'x' should have 2 or more columns.
#' @param y response variable. Quantitative for 'family="gaussian"'. For 'family="binomial"' should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For 'family="multinomial"', can be a 'nc>=2' level factor, or a matrix with 'nc' columns of counts or proportions. For either '"binomial"' or '"multinomial"', if 'y' is presented as a vector, it will be coerced into a factor. For 'family="cox"', preferably a 'Surv' object from the survival package: see Detail section for more information. For 'family="mgaussian"', 'y' is a matrix of quantitative responses.
#' @param groups A vector of length nobs indicating to which group each observation belongs. For data with k groups, groups should be coded as integers 1 through k. 
#' @param alpha The pretrained lasso hyperparameter, with \eqn{0\le\alpha\le 1}.
#' @param family Either a character string representing one of the built-in families, or else a 'glm()' family object. For more information, see Details section below or the documentation for response type (above).
#' @param type.measure loss to use for cross-validation within each individual, overall, or pretrained lasso model. Currently five options, not all available for all models. The default is 'type.measure="deviance"', which uses squared-error for gaussian models (a.k.a 'type.measure="mse"' there), deviance for logistic and poisson regression, and partial-likelihood for the Cox model. 'type.measure="class"' applies to binomial and multinomial logistic regression only, and gives misclassification error. 'type.measure="auc"' is for two-class logistic regression only, and gives area under the ROC curve. 'type.measure="mse"' or 'type.measure="mae"' (mean absolute error) can be used by all models except the '"cox"'; they measure the deviation from the fitted mean to the response. 'type.measure="C"' is Harrel's concordance measure, only available for 'cox' models.
#' @param use.case The type of grouping observed in the data. Can be one of "inputGroups" or "targetGroups".
#' @param overall.lambda The choice of lambda to be used by the overall model when defining the offset and penalty factor for pretrained lasso. Defaults to "lambda.1se", but "lambda.min" is another good option. If known in advance, can alternatively supply a numeric value.
#' @param fitall An optional cv.glmnet object specifying the overall model.
#' @param fitind An optional list of cv.glmnet objects specifying the individual models.
#' @param lambda Optional user-supplied lambda sequence; default is 'NULL', and 'glmnet' chooses its own sequence. Note that this is done for the full model (master sequence), and separately for each fold. The fits are then alligned using the master sequence (see the 'alignment' argument for additional details). Adapting 'lambda' for each fold leads to better convergence. When 'lambda' is supplied, the same sequence is used everywhere, but in some GLMs can lead to convergence issues.
#' @param nfolds Number of folds for CV (default is 10). Although \code{nfolds}can be as large as the sample size (leave-one-out CV), it is not recommended for large datasets. Smallest value allowable is \code{nfolds = 3}.
#' @param foldid An optional vector of values between 1 and \code{nfold} identifying what fold each observation is in. If supplied, \code{nfold} can be missing.
#' @param standardize Should the predictors be standardized before fitting (default is TRUE).
#' @param verbose If \code{verbose=1}, print a statement showing which model is currently being fit with \code{cv.glmnet}.
#' @param weights observation weights. Default is 1 for each observation.
#' @param penalty.factor Separate penalty factors can be applied to each coefficient. This is a number that multiplies 'lambda' to allow differential shrinkage. Can be 0 for some variables,  which implies no shrinkage, and that variable is always included in the model. Default is 1 for all variables (and implicitly infinity for variables listed in 'exclude'). For more information, see \code{?glmnet}. For pretraining, the user-supplied penalty.factor will be multiplied by the penalty.factor computed by the overall model.
#' @param \dots Additional arguments to be passed to the cv.glmnet function. Some notable choices are \code{"trace.it"} and \code{"parallel"}. If \code{trace.it = TRUE}, then a progress bar is displayed for each call to \code{cv.glmnet}; useful for big models that take a long time to fit. If \code{parallel = TRUE}, use parallel \code{foreach} to fit each fold.  Must register parallel before hand, such as \code{doMC} or others. Importantly, \code{"ptLasso"} does not support the arguments \code{"intercept"}, \code{"offset"}, \code{"fit"} and \code{"check.args"}.
#'
#' 
#'
#' @return An object of class \code{"ptLasso"}, which is a list with the ingredients of the cross-validation fit.
#' \item{call}{The call that produced this object.}
#' \item{k}{The number of groups.}
#' \item{alpha}{The value of alpha used for pretraining.}
#' \item{group.levels}{IDs for all of the groups used in training.}
#' \item{fitall}{A fitted \code{cv.glmnet} object trained using the full data.}
#' \item{fitpre}{A list of fitted (pretrained) \code{cv.glmnet} objects, one trained with each data group.}
#' \item{fitind}{A list of fitted \code{cv.glmnet} objects, one trained with each group.}
#' \item{fitall.lambda}{Lambda used with fitall, to compute the offset for pretraining. Used for prediction.}
#' \item{y.mean}{Gaussian outcome only; mean of y for the training data, used for prediction.}
#' 
#' @examples
#' # Gaussian
#' 
#' scommon     = 10                  # Number of common important  features
#' sindiv      = c(50,40,20,10,10)   # Number of individual important features for each group
#' class.sizes = c(100,80,60,30,30)  # Size of each group
#' k           = length(class.sizes) # Number of groups
#' n           = sum(class.sizes)    # Total dataset size
#' p           = 200                 # Number of features 
#' beta.common = rep(2.5, k)         # common coefficients 
#' beta.indiv  = rep(1, k)           # individual coefficients
#' intercepts  = rep(0, k)           # group-specific intercepts
#' sigma       = 20                  # size of noise added
#'
#' # Train data
#' out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
#'              class.sizes=class.sizes, beta.common=beta.common, beta.indiv=beta.indiv,
#'              intercepts=intercepts, sigma=sigma, outcome="gaussian")
#' 
#' x=out$x; y=out$y; groups=out$groups
#'
#' # Test data
#' outtest=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
#'                  class.sizes=class.sizes, beta.common=beta.common, beta.indiv=beta.indiv,
#'                  intercepts=intercepts, sigma=sigma, outcome="gaussian")
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#' 
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' # plot(fit) to see all of the cv.glmnet models trained
#' predict(fit, xtest, groupstest, ytest=ytest)
#' 
#' @import glmnet Matrix
#' @export
#' @seealso \code{\link{glmnet}, \link{sparsenet}}
#' @references Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1-22.
#' Mazumder, Rahul, Jerome H. Friedman, and Trevor Hastie. "Sparsenet: Coordinate descent with nonconvex penalties." Journal of the American Statistical Association 106.495 (2011): 1125-1138.
#'
#' 
# Note: cv doesn't have to check everything that ptLasso checks
ptLasso=function(x,y,groups,alpha=0.5,family=c("gaussian", "multinomial", "binomial","cox"),
                 type.measure=c("default", "mse", "mae", "auc","deviance","class", "C"),
                 use.case=c("inputGroups","targetGroups"),
                 overall.lambda = "lambda.1se",
                 #fit.method = c("glmnet", "sparsenet"),
                 lambda=NULL, foldid=NULL,
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

    if(!(family %in% names(this.call))) this.call$family = family
    if(!(type.measure %in% names(this.call))) this.call$type.measure = type.measure
    if(!(use.case %in% names(this.call))) this.call$use.case = use.case
    
    np=dim(x)
    ##check dims
    if(is.null(np)|(np[2]<=1))stop("x should be a matrix with 2 or more columns")
    nobs=as.integer(np[1])
    nvars=as.integer(np[2])

    k = length(table(groups))
    
    ####################################################################################
    # Begin error checking:
    ####################################################################################

    #if((fit.method == "sparsenet") & (family != "gaussian")) stop("sparsenet is only available for the Gaussian family.")
    
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
        if(!("cv.glmnet" %in% class(fitall))) stop("fitall must be a cv.glmnet object.")
     }
    if(!is.null(fitind) & !(all(sapply(fitind, function(mm) "cv.glmnet" %in% class(mm))))) stop("fitind must be a list of cv.glmnet objects.")

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
    ####################################################################################
    # End error checking:
    ####################################################################################

    model = cv.glmnet
    #if(fit.method == "sparsenet") model = cv.sparsenet
        
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

    nvars = ncol(x)
    x.mean = rep(0.0, times = nvars)
    x.sds  = rep(1.0, times = nvars)
    y.mean = NULL
    if(is.null(weights)) weights = rep(1.0, times = nrow(x))
    if(standardize){
        if(!inherits(x, "sparseMatrix")){
            meansd <- glmnet:::weighted_mean_sd(x, weights)
            x.mean <- meansd$mean
            x.sds  <- meansd$sd

            x <- scale(x, x.mean, x.sds)
        }
        y.mean = 0
        if(family == "gaussian"){ 
            y.mean = mean(y)
            y = y - y.mean
        }
    }

    ####################################################################################
    # Fit overall model 
    ####################################################################################

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
            fitall = model(cbind(onehot.groups, x), y,
                               family=family,
                               foldid=foldid, 
                               intercept=TRUE,
                               lambda=lambda,
                               type.measure=type.measure,
                               standardize=FALSE,
                               penalty.factor=overall.pf,
                               keep=TRUE,
                               weights=weights,
                               ...)
        } else if(family == "cox") {
            fitall = model(cbind(onehot.groups, x), y,
                               family=family,
                               foldid=foldid,  
                               lambda=lambda,
                               type.measure=type.measure,
                               standardize=FALSE,
                               penalty.factor=overall.pf,
                               keep=TRUE,
                               weights=weights,
                               ...)     
        } else if(use.case == "targetGroups") {
            type.multinomial = "grouped"
            if("type.multinomial" %in% names(list(...))) type.multinomial = list(...)$type.multinomial
            fitall = model(x,y,
                               family=family,
                               foldid=foldid,  
                               lambda=lambda,
                               type.measure=type.measure,
                               standardize=FALSE,
                               penalty.factor=overall.pf,
                               type.multinomial = type.multinomial,
                               keep=TRUE,
                               weights=weights,
                               ...)     
        }
    }

    if(overall.lambda == "lambda.min") lamhat = fitall$lambda.min
    if(overall.lambda == "lambda.1se") lamhat = fitall$lambda.1se
    if(is.numeric(overall.lambda)) lamhat = overall.lambda

    if(use.case=="inputGroups"){
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
    } else if(use.case=="targetGroups"){
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

    ####################################################################################
    # Fit individual models
    ####################################################################################
    fitind.is.null = is.null(fitind)

    if(verbose & fitind.is.null) cat("Fitting individual models",fill=TRUE)
    
    if(use.case=="inputGroups"){
        if(fitind.is.null) fitind=vector("list",k) #bhatInd
        
        for(kk in 1:k){
            train.ix = groups == kk
            
            # individual model predictions
            if(fitind.is.null){
                if(family!="cox") { 
                    fitind[[kk]]=model(x[train.ix,], y[train.ix],
                                           family=family,
                                           standardize = FALSE,
                                           type.measure=type.measure,
                                           foldid=foldid2[[kk]],
                                           intercept=intercept,
                                           penalty.factor=penalty.factor,
                                           keep=TRUE,
                                           weights=weights[train.ix],
                                           ...)
                } else if(family=="cox") {
                    fitind[[kk]]=model(x[train.ix,], y[train.ix, ],
                                           family=family,
                                           standardize = FALSE,
                                           type.measure=type.measure,
                                           foldid=foldid2[[kk]],
                                           keep=TRUE,
                                           weights=weights[train.ix],
                                           ...)
                }
            }
        }
    }
   
    if(use.case=="targetGroups"){
        #if(fitind.is.null) fitind=bhatInd=vector("list",k)
        #supind=matrix(NA,k,p)
        
        for(kk in 1:k){
            if(fitind.is.null){
                yy = rep(0, nrow(x))
                yy[y == kk]=1
                
                fitind[[kk]] = model(x,yy,
                                         family="binomial",
                                         foldid=foldid,
                                         type.measure=type.measure,
                                         penalty.factor=penalty.factor,
                                         standardize=FALSE,
                                         keep=TRUE,
                                         weights=weights,
                                         ...)
           }
           lamhat2 = fitind[[kk]]$lambda.min
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
                     offset = (1-alpha) * fitall$fit.preval[train.ix, , fitall$lambda == lamhat]
                 } else {
                     offset = (1-alpha) * fitall$fit.preval[train.ix, fitall$lambda == lamhat]
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

                 if(family!="cox") fitpre[[kk]] = model(x[train.ix,],
                                                            y[train.ix],
                                                            family=family, 
                                                            offset=offset,
                                                            intercept=intercept,
                                                            type.measure=type.measure,
                                                            standardize=FALSE,
                                                            foldid=foldid2[[kk]],
                                                            keep=TRUE,
                                                            penalty.factor=pf,
                                                            weights=weights[train.ix],
                                                            ...)
                 if(family=="cox") fitpre[[kk]] = model(x[train.ix,],
                                                            y[train.ix,],
                                                            family=family, 
                                                            offset=offset,
                                                            type.measure=type.measure,
                                                            penalty.factor=pf,
                                                            standardize=FALSE,
                                                            foldid=foldid2[[kk]],
                                                            keep=TRUE,
                                                            weights=weights[train.ix],
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

                 fitpre[[kk]] = model(x,yy,
                                          family="binomial", 
                                          offset=myoffset,
                                          penalty.factor=pf,
                                          foldid=foldid,
                                          standardize=FALSE,
                                          keep=TRUE,
                                          type.measure=type.measure,
                                          weights=weights,
                                          ...)
                 
                 
                 
             }
         }
     }


    # Return coefficients to original scale.
    # Start with the overall model.
    if(standardize){
        if(fitall.is.null){
            if( (use.case == "inputGroups") & (family != "multinomial") ){
                beta <- fitall$glmnet.fit$beta; a0 <- fitall$glmnet.fit$a0;
                ix <- k:(k+p-1)                  # the first k-1 are group indicators...
                if(family == "cox") ix = ix + 1  # except for the cox model, which has k group indicators.
                
                beta[ix, ] <- beta[ix, ]/x.sds
                a0 <- a0 - Matrix::colSums( beta[ix, ] * x.mean )

                fitall$glmnet.fit$beta <- beta; fitall$glmnet.fit$a0 <- a0
            } else if( (use.case == "inputGroups") & (family == "multinomial") ) {
                ix <- k:(k+p-1)
                for(kk in 1:length(fitall$glmnet.fit$beta)){
                    beta <- fitall$glmnet.fit$beta[[kk]]; a0 <- fitall$glmnet.fit$a0[kk, ];
                    
                    beta[ix, ] <- beta[ix, ]/x.sds
                    a0 <- a0 - Matrix::colSums(beta[ix, ] * x.mean)

                    fitall$glmnet.fit$beta[[kk]] <- beta; fitall$glmnet.fit$a0[kk, ] <- a0
                }
            } else if(use.case == "targetGroups") {
                for(kk in 1:k){
                    beta <- fitall$glmnet.fit$beta[[kk]]; a0 <- fitall$glmnet.fit$a0[kk, ];
                    
                    beta <- beta/x.sds
                    a0 <- a0 - Matrix::colSums(beta * x.mean)

                    fitall$glmnet.fit$beta[[kk]] <- beta; fitall$glmnet.fit$a0[kk, ] <- a0
                }
            }
        }

        # Now, the individual and pretrained models:
        if( (use.case == "inputGroups") & (family == "multinomial") ){
            for(kk in 1:k){
                # Individual
                if(fitind.is.null){
                    for(oc in 1:length(table(y))){
                        beta <- fitind[[kk]]$glmnet.fit$beta[[oc]];
                        a0   <- fitind[[kk]]$glmnet.fit$a0[oc, ];

                        beta <- beta/x.sds
                        a0   <- a0 - Matrix::colSums(beta * x.mean)

                        fitind[[kk]]$glmnet.fit$beta[[oc]] = beta
                        fitind[[kk]]$glmnet.fit$a0[oc, ]   = a0
                    }
                }
                # Pretrained:
                for(oc in 1:length(table(y))){
                    beta <- fitpre[[kk]]$glmnet.fit$beta[[oc]];
                    a0   <- fitpre[[kk]]$glmnet.fit$a0[oc, ];

                    beta <- beta/x.sds
                    a0   <- a0 - Matrix::colSums(beta * x.mean)

                    fitpre[[kk]]$glmnet.fit$beta[[oc]] = beta
                    fitpre[[kk]]$glmnet.fit$a0[oc, ]   = a0
                }
            }
        } else {
            for(kk in 1:k){
                # Individual
                if(fitind.is.null){
                    beta <- fitind[[kk]]$glmnet.fit$beta;
                    a0   <- fitind[[kk]]$glmnet.fit$a0;

                    beta <- beta/x.sds
                    a0   <- a0 - Matrix::colSums(beta * x.mean)

                    fitind[[kk]]$glmnet.fit$beta = beta
                    fitind[[kk]]$glmnet.fit$a0   = a0
                }
                # Pretrained:
                beta <- fitpre[[kk]]$glmnet.fit$beta;
                a0   <- fitpre[[kk]]$glmnet.fit$a0;

                beta <- beta/x.sds
                a0   <- a0 - Matrix::colSums(beta * x.mean)

                fitpre[[kk]]$glmnet.fit$beta = beta
                fitpre[[kk]]$glmnet.fit$a0   = a0
            }
        }
    }

    
    
    out=enlist(
               # Info about the initial call:
               call=this.call,
               k, alpha, group.levels,
               
               # Fitted models
               fitall, lamhat, 
               fitind, fitpre
    )
    if(family == "gaussian") out$y.mean = y.mean
    class(out)="ptLasso"
    return(out)

}


    

  
 
