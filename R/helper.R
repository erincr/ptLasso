
#' Print the cv.ptLasso object. 
#'
#'
#' @aliases print.cv.ptLasso 
#' @param x fitted \code{"cv.ptLasso"} object.
#' @param digits number of digits to display.
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
print.cv.ptLasso=function (x, digits = max(3, getOption("digits") - 3)) 
{
    cat("\nCall: ", deparse(x$call), "\n\n", fill = 50)
    
    if("errpre" %in% names(x)){
        cat("type.measure: ", x$call$type.measure, "\n\n")
        disp = rbind(c(NA, x$errall), rbind(x$errpre, c(NA, x$errind)))
        rownames(disp) = c("Overall", rep("Pretrain", nrow(x$errpre)), "Individual")
        colnames(disp)[1] = "alpha"
        cat("",fill=TRUE)
        print(disp, digits = digits, na.print="")        
     }

    cat("",fill=TRUE)
    cat(c("alphahat (fixed) =",x$alphahat),fill=TRUE)
    cat("alphahat (varying):\n")
    alpha.disp = x$varying.alphahat
    names(alpha.disp) = paste0("group", 1:length(x$varying.alphahat))
    print(alpha.disp)
    #cat(paste(x$varying.alphahat, collapse=", "),fill=TRUE)
    
}

#' Print the ptLasso object. 
#'
#'
#' @aliases print.ptLasso 
#' @param x fitted \code{"ptLasso"} object.
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
print.ptLasso=function (x) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")
}




#' Print the predict.ptLasso object. 
#'
#'
#' @aliases print.predict.ptLasso 
#' @param x output of predict called with a ptLasso object.
#' @param digits number of digits to display.
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
print.predict.ptLasso=function (x, digits = max(3, getOption("digits") - 3)) 
{
    cat("\nCall: ", deparse(x$call), "\n\n", fill = 50)
  
    cat(c("alpha = ", x$alpha, "\n"), fill=TRUE)

    if("errpre" %in% names(x)){
        #cat("type.measure: ", x$type.measure, "\n\n")
        disp = rbind(x$errall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat("Performance (", x$type.measure, "):", "\n", sep="")
        cat("", fill=TRUE)
        print(disp, digits = digits, na.print="")   
        cat("\n")
     }

    cat("Support size:\n")
    disp.support = matrix(c(length(x$supall),
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
#' @param digits number of digits to display.
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
print.predict.cv.ptLasso=function (x, digits = max(3, getOption("digits") - 3)) 
{
    cat("\nCall: ", deparse(x$call), "\n\n", fill = 50)

    if(length(x$alpha) > 1){
          alpha.disp = x$alpha
          names(alpha.disp) = paste0("group", 1:length(x$alpha))
          cat("alpha:\n")
          print(alpha.disp)
        } else {
          cat(c("alpha =",x$alpha), fill=TRUE)
        }
    cat("\n",fill=TRUE)
    
    if("errpre" %in% names(x)){
        disp = rbind(x$errall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat("Performance (", x$type.measure, "):", "\n", sep="")
        print(disp, digits = digits)  
        cat("\n",fill=TRUE)

     }
        
    cat("Support size:\n")
    disp.support = matrix(c(length(x$supall),
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
