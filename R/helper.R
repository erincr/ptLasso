#
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
#'
#'
#' @method print cv.ptLasso
#' @export
#'
#'
print.cv.ptLasso=function (x, digits = max(3, getOption("digits") - 3)) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")
    
    if("errpre" %in% names(x)){
        cat("type.measure: ", x$call$type.measure, "\n\n")
        disp = rbind(c(NA, x$errall), rbind(x$errpre, c(NA, x$errind)))
        rownames(disp) = c("Overall", rep("Pretrain", nrow(x$errpre)), "Individual")
        colnames(disp)[1] = "alpha"
        cat("",fill=TRUE)
        print(disp, digits = digits, na.print="")        
     }

    cat("",fill=TRUE)
    cat(c("alphahat=",x$alphahat),fill=TRUE)
    
}

#
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
#'
#'
#' @method print ptLasso
#' @export
#'
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
#'
#'
#' @method print predict.ptLasso
#' @export
print.predict.ptLasso=function (x, digits = max(3, getOption("digits") - 3)) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")

    if("errpre" %in% names(x)){
        cat("type.measure: ", x$call$type.measure, "\n\n")
        disp = rbind(x$errall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat(c("alpha =", x$alpha), fill=TRUE)
        cat("", fill=TRUE)
        print(disp, digits = digits, na.print="")           
     }
        
     else {
        # Anything better we can do here?
        cat("\npred$yhatpre\n")
        print(x$yhatpre)
   }

    
}


#' Print the predict.ptLasso object. 
#'
#'
#' @aliases print.predict.cv.ptLasso 
#' @param x output of predict called with a ptLasso object.
#' @param digits number of digits to display.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer:
#' Erin Craig <erincr@@stanford.edu>
#' @seealso \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#'
#'
#' @method print predict.cv.ptLasso
#' @export
#'
print.predict.cv.ptLasso=function (x, digits = max(3, getOption("digits") - 3)) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")

    if("errpre" %in% names(x)){
        cat("type.measure: ", x$type.measure, "\n\n")
        disp = rbind(x$errall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat(c("alpha =",x$alpha),fill=TRUE)
        cat("",fill=TRUE)
        print(disp, digits = digits)        
     }
        
     else {
        # Anything better we can do here?
        cat("\npred$yhatpre\n")
        print(x$yhatpre)
   }
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
