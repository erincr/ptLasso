
#' Print the cv.ptLasso object. 
#'
#'
#' @aliases print.cv.ptLasso 
#' @param x fitted \code{"cv.ptLasso"} object.
#' @param \dots other arguments to pass to the print function.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' print(cvfit)
#'
#' @method print cv.ptLasso
#' @export
print.cv.ptLasso=function (x, ...) 
{
    cat("\nCall: ", deparse(x$call), "\n\n", fill = 50)

    digits = max(3, getOption("digits") - 3)
    
    if("errpre" %in% names(x)){
        cat("type.measure: ", x$call$type.measure, "\n\n")
        disp = rbind(c(NA, x$erroverall), rbind(x$errpre, c(NA, x$errind)))
        rownames(disp) = c("Overall", rep("Pretrain", nrow(x$errpre)), "Individual")
        colnames(disp)[1] = "alpha"
        cat("",fill=TRUE)
        print(disp, digits = digits, na.print="")        
     }

    cat("",fill=TRUE)
    cat(c("alphahat (fixed) =",x$alphahat),fill=TRUE)
    cat("alphahat (varying):\n")
    alpha.disp = x$varying.alphahat
    if("nresps" %in% names(x$fit[[1]])){
        names(alpha.disp) = colnames(x$errpre)[grepl("response", colnames(x$errpre))]
    } else {
        names(alpha.disp) = colnames(x$errpre)[grepl("group", colnames(x$errpre))]
    }
    print(alpha.disp)
    #cat(paste(x$varying.alphahat, collapse=", "),fill=TRUE)
    
}

#' Print the ptLasso object. 
#'
#'
#' @aliases print.ptLasso 
#' @param x fitted \code{"ptLasso"} object.
#' @param \dots other arguments to pass to the print function.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' print(fit)
#'
#' @method print ptLasso
#' @export
#'
print.ptLasso=function (x, ...) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")
}




#' Print the predict.ptLasso object. 
#'
#'
#' @aliases print.predict.ptLasso 
#' @param x output of predict called with a ptLasso object.
#' @param \dots other arguments to pass to the print function.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso} and \code{predict.ptLasso}.
#' @keywords models regression classification
#' @examples
#' # Train data
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' # Test data
#' outtest = gaussian.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#'
#' fit = ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' pred = predict(fit, xtest, groupstest, ytest=ytest, s="lambda.min")
#' print(pred)
#'
#' # If ytest is not supplied, just prints the pretrained predictions.
#' pred = predict(fit, xtest, groupstest, s="lambda.min")
#' print(pred)
#' 
#' @method print predict.ptLasso
#' @export
print.predict.ptLasso=function (x, ...) 
{
    digits = max(3, getOption("digits") - 3)
    
    cat("\nCall: ", deparse(x$call), "\n\n", fill = 50)
  
    cat(c("alpha = ", x$alpha, "\n"), fill=TRUE)

    if("errpre" %in% names(x)){
        #cat("type.measure: ", x$type.measure, "\n\n")
        disp = rbind(x$erroverall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat("Performance (",  yaxis.name(x$type.measure), "):", "\n", sep="")
        cat("", fill=TRUE)
        print(disp, digits = digits, na.print="")   
        cat("\n")
     }

    cat("Support size:\n")
    disp.support = matrix(c(length(x$supoverall),
                            paste0(length(x$suppre.common) + length(x$suppre.individual),
                                   " (", length(x$suppre.common), " common + ",
                                   length(x$suppre.individual), " individual)"),
                            length(x$supind)), ncol=1)
    rownames(disp.support) = c("Overall", "Pretrain", "Individual")
    colnames(disp.support) = ""
    print(noquote(disp.support))
    
}


#' Print the predict.cv.ptLasso object. 
#'
#' @aliases print.predict.cv.ptLasso 
#' @param x output of predict called with a ptLasso object.
#' @param \dots other arguments to pass to the print function.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' # Train data
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group;
#'
#' # Test data
#' outtest = gaussian.example.data()
#' xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups
#'
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
#' print(pred)
#'
#' # If ytest is not supplied, just prints the pretrained predictions.
#' pred = predict(cvfit, xtest, groupstest, s="lambda.min")
#' print(pred)
#'
#' @method print predict.cv.ptLasso
#' @export
print.predict.cv.ptLasso=function (x, ...) 
{
    digits = max(3, getOption("digits") - 3)
    
    cat("\nCall: ", deparse(x$call), "\n\n", fill = 50)

    if(length(x$alpha) > 1){
          alpha.disp = x$alpha
          names(alpha.disp) = paste0("group_", 1:length(x$alpha))
          cat("alpha:\n")
          print(alpha.disp)
        } else {
          cat(c("alpha =",x$alpha), fill=TRUE)
        }
    cat("\n",fill=TRUE)
    
    if("errpre" %in% names(x)){
        disp = rbind(x$erroverall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat("Performance (", yaxis.name(x$type.measure), "):", "\n", sep="")
        print(disp, digits = digits)  
        cat("\n",fill=TRUE)

     }
    
    cat("Support size:\n")
    disp.support = matrix(c(length(x$supoverall),
                          #paste0(length(x$suppre.common), " + ", length(x$suppre.individual), " (common + individual)"),
                          paste0(length(x$suppre.common) + length(x$suppre.individual),
                                 " (", length(x$suppre.common), " common + ",
                                 length(x$suppre.individual), " individual)"),
                          length(x$supind)), ncol=1)
    rownames(disp.support) = c("Overall", "Pretrain", "Individual")
    colnames(disp.support) = ""
    print(noquote(disp.support))
}


#' @noRd
enlist <-
function (...) 
{
    result <- list(...)
    if ((nargs() == 1) & is.character(n <- result[[1]])) {
        result <- as.list(seq(n))
        names(result) <- n
        for (i in n) result[[i]] <- get(i)
    }
    else {
        n <- sys.call()
        n <- as.character(n)[-1]
        if (!is.null(n2 <- names(result))) {
            which <- n2 != ""
            n[which] <- n2[which]
        }
        names(result) <- n
    }
    result
}

#' @noRd
balanced.folds <- function(y, nfolds = min(min(table(y)), 10)) {
   totals <- table(y)
   fmax <- max(totals)
   nfolds <- min(nfolds, fmax)     
   nfolds= max(nfolds, 2)
   
   # makes no sense to have more folds than the max class size
   folds <- as.list(seq(nfolds))
   yids <- split(seq(y), y)
   
         # nice we to get the ids in a list, split by class
###Make a big matrix, with enough rows to get in all the folds per class
   bigmat <- matrix(NA, ceiling(fmax/nfolds) * nfolds, length(totals))
   for(i in seq(totals)) {

     if(length(yids[[i]])>1){bigmat[seq(totals[i]), i] <- sample(yids[[i]])}
     if(length(yids[[i]])==1){bigmat[seq(totals[i]), i] <- yids[[i]]}

   }
   smallmat <- matrix(bigmat, nrow = nfolds)# reshape the matrix
### Now do a clever sort to mix up the NAs
   smallmat <- permute.rows(t(smallmat))   ### Now a clever unlisting
         # the "clever" unlist doesn't work when there are no NAs
         #       apply(smallmat, 2, function(x)
         #        x[!is.na(x)])
   res <-vector("list", nfolds)
   for(j in 1:nfolds) {
     jj <- !is.na(smallmat[, j])
     res[[j]] <- smallmat[jj, j]
   }
   return(res)
 }

#' @noRd
permute.rows <- function(x)
{
        dd <- dim(x)
        n <- dd[1]
        p <- dd[2]
        mm <- runif(length(x)) + rep(seq(n) * 10, rep(p, n))
        matrix(t(x)[order(mm)], n, p, byrow = TRUE)
}
