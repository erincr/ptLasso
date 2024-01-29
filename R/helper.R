#
#' Print the cv.ptLasso object. 
#'
#'
#' @aliases print.cv.ptLasso 
#' @param x fitted \code{"cv.ptLasso"} or \code{"predict.cv.ptLasso"} object.
#' @param \dots Other graphical parameters to plot
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

print.cv.ptLasso=function (x, digits = max(3, getOption("digits") - 3), ...) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")
    
    if("errpre" %in% names(x)){
        disp = rbind(c(NA, x$errall), rbind(x$errpre, c(NA, x$errind)))
        rownames(disp) = c("Overall", rep("Pretrain", nrow(x$errpre)), "Individual")

        cat("",fill=TRUE)
        print(disp, digits = digits, na.print="")        
     }

    cat("",fill=TRUE)
    cat(c("alphahat=",x$alphahat),fill=TRUE)
    
}

# TODO!
# What to print?
print.ptLasso=function (x, digits = max(3, getOption("digits") - 3), ...) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")
}


print.predict.ptLasso=function (x, digits = max(3, getOption("digits") - 3), ...) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")

    if("errpre" %in% names(x)){
        disp = rbind(x$errall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat(c("alpha=", x$alpha), fill=TRUE)
        cat("", fill=TRUE)
        print(disp, digits = digits, na.print="")           
     }
        
     else {
        # Anything better we can do here?
        cat("\npred$yhatpre\n")
        print(x$yhatpre)
   }

    
}

print.predict.cv.ptLasso=function (x, digits = max(3, getOption("digits") - 3), ...) 
{
    cat("\nCall: ", deparse(x$call), "\n\n")

    if("errpre" %in% names(x)){
        disp = rbind(x$errall, rbind(x$errpre,x$errind))
        rownames(disp) = c("Overall", "Pretrain", "Individual")

        cat(c("alpha=",x$alpha),fill=TRUE)
        cat("",fill=TRUE)
        print(disp, digits = digits)        
     }
        
     else {
        # Anything better we can do here?
        cat("\npred$yhatpre\n")
        print(x$yhatpre)
   }

    
}



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
