#' Get the y axis label
#' @noRd
yaxis.name = function(x){
    if(x == "class")      return("Misclassification error")
    if(x == "mse")        return("Mean squared error")
    if(x == "mae")        return("Mean absolute error")
    if(x == "deviance")   return("Deviance")
    if(x == "auc")        return("AUC")
    if(x == "C")          return("C-index")
}
 
#' Plot the cross-validation curve produced by cv.ptLasso, as a function of the \code{alpha} values used. 
#'
#' A plot is produced, and nothing is returned.
#'
#' @aliases plot.cv.ptLasso 
#' @param x Fitted \code{"cv.ptLasso"} object.
#' @param plot.alphahat If \code{TRUE}, show a dashed vertical line indicating the single value of alpha that maximized overall cross-validated performance.
#' @param \dots Other graphical parameters to plot.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group
#' 
#' cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
#' plot(cvfit) 
#'
#' @import ggplot2 gridExtra
#' @method plot cv.ptLasso
#' @export
#'
#'
plot.cv.ptLasso = function(x, plot.alphahat = TRUE, ...){
    if(x$call$use.case %in% c("inputGroups", "multiresponse"))  ggplot.ptLasso.inputGroups(x,  plot.alphahat = plot.alphahat, y.label = yaxis.name(x$call$type.measure), ...)
    if(x$call$use.case == "targetGroups") ggplot.ptLasso.targetGroups(x, plot.alphahat = plot.alphahat, y.label = yaxis.name(x$call$type.measure), ...)
}


overall.color    <- "#E9C46A"
individual.color <- "#E76F51"
pretrain.color   <- "#2A9D8F"
##################
# GGPLOT
##################
#' Plot function for target grouped data
#' @noRd
ggplot.ptLasso.targetGroups=function(x, y.label, plot.alphahat = FALSE,...){

    err.pre = x$errpre
    err.pan = x$erroverall
    err.ind = x$errind
    n.alpha = nrow(err.pre)

    if(is.null(x$suppre)){
        suppre = sapply(x$fit, function(ff) length(get.pretrain.support(ff, commonOnly = FALSE)))
        supind = length(get.individual.support(x, commonOnly = FALSE))
        suppan = length(get.overall.support(x))
    } else {
        suppre = sapply(x$suppre, function(ff) length(ff))
        supind = length(x$supind)
        suppan = length(x$supall)
    }

    k = ncol(err.pre) - 3

    ylim = range(c(err.pre[, "overall"], err.pan["overall"], err.ind["overall"])) 
    nudge = .1 * (ylim[2] - ylim[1])
    ylim[2] = ylim[2] + nudge

    n.alpha = nrow(err.pre)

    forplot = data.frame(
        "alpha"   = c(err.pre[,"alpha"], err.pre[,"alpha"], err.pre[,"alpha"]),
        "overall" = c(err.pre[, "overall"], rep(err.ind["overall"], n.alpha), rep(err.pan["overall"], n.alpha)),
        "model"   = c(rep("Pretrain", n.alpha), rep("Individual", n.alpha), rep("Overall (grouped)", n.alpha))
    )
    ylims = range(forplot$overall)
    nudge = .1 * (ylims[2] - ylims[1])
    ylims[2] = ylims[2] + nudge
    ylims[1] = ylims[1] - nudge
    
    plot1 <- ggplot() +
        geom_line(aes(x=forplot$alpha, y=forplot$overall,
                      group = forplot$model, color = forplot$model)) +
        geom_text(aes(x=.2, y=err.pan["overall"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
        geom_text(aes(x=.2, y=err.ind["overall"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
        scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
        labs(x = expression(alpha), y = y.label, color = "", subtitle=paste0(as.character(k)," class problem")) +
        ylim(ylims[1], ylims[2]) +
        theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall (grouped)", "Pretrain", "Individual")) +
        guides(color="none")

    group.cols = colnames(err.pre)[grepl("group", colnames(err.pre))]
    sum.of.indiv.pre = rowSums(err.pre[, group.cols])
    sum.of.indiv.ind = sum(err.ind[group.cols])
    forplot2 <- data.frame(
        "alpha"          = c(err.pre[,"alpha"], err.pre[,"alpha"]),
        "individualsum"  = c(sum.of.indiv.pre, rep(sum.of.indiv.ind, n.alpha)),
        "model"          = factor(c(rep("Pretrain", n.alpha), rep("Individual", n.alpha)), levels = c("Pretrain", "Individual", "Overall\n(grouped)"))
    )
    ylims2 = range(forplot2$individualsum)
    nudge = .1 * (ylims2[2] - ylims2[1])
    ylims2[2] = ylims2[2] + nudge

    plot2 <- ggplot() +
        geom_line(aes(x=forplot2$alpha, y=forplot2$individualsum,
                      group = forplot2$model, color = forplot2$model)) +
        geom_text(aes(x=.2, y=sum.of.indiv.ind, label=as.character(supind), vjust=-1), size=3, color="#666666") +
        scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
        labs(x = expression(alpha), y = y.label, color = "", subtitle="Sum of individual one vs. rest problems") +
        ylim(ylims2[1], ylims2[2]) +
        theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall\n(grouped)", "Pretrain", "Individual"), drop = FALSE) 

    if(plot.alphahat) {
        plot1 = plot1 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
        plot2 = plot2 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
    }
    
    gridExtra::grid.arrange(plot1, plot2, ncol=2, widths=c(.75, 1))
}


#' Plot function for input grouped data
#' @noRd
ggplot.ptLasso.inputGroups=function(x, y.label, plot.alphahat = FALSE,...){

    nm = "group"
    if(x$call$use.case == "multiresponse") nm = "response"
    
    k = length(x$fit[[1]]$fitind)
    err.pre = x$errpre
    err.pan = x$erroverall
    err.ind = x$errind
    
    suppre = sapply(x$fit, function(ff) length(get.pretrain.support(ff, commonOnly = FALSE)))
    supind = length(get.individual.support(x, commonOnly = FALSE))
    suppan = length(get.overall.support(x))
    
    n.alpha = nrow(err.pre)

    forplot = data.frame(
        "alpha"   = c(err.pre[,"alpha"], err.pre[,"alpha"], err.pre[,"alpha"]),
        "overall" = c(err.pre[, "overall"], rep(err.ind["overall"], n.alpha), rep(err.pan["overall"], n.alpha)),
        "model"   = c(rep("Pretrain", n.alpha), rep("Individual", n.alpha), rep("Overall", n.alpha))
    )
    ylims = range(forplot$overall)
    nudge = .1 * (ylims[2] - ylims[1])
    ylims[2] = ylims[2] + nudge
    ylims[1] = ylims[1] - nudge

    plot1 <- ggplot() +
        geom_line(aes(x=forplot$alpha, y=forplot$overall,
                      group = forplot$model, color = forplot$model)) +
        geom_text(aes(x=.2, y=err.pan["overall"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
        geom_text(aes(x=.2, y=err.ind["overall"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
        scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
        labs(x = expression(alpha), y = y.label, color = "", title=paste0(as.character(k), " ", nm, " problem")) +
        ylim(ylims[1], ylims[2]) +
        theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall", "Pretrain", "Individual"))
    
    if(y.label == "Mean squared error") {
        # Average of group-specific problems is the same as the overall model
        print(plot1)
        return()
    }
    plot1 = plot1 + guides(color="none")
    
    forplot2 = data.frame(
        "alpha"   = c(err.pre[,"alpha"], err.pre[,"alpha"], err.pre[,"alpha"]),
        "overall" = c(err.pre[, "mean"], rep(err.ind["mean"], n.alpha), rep(err.pan["mean"], n.alpha)),
        "model"   = c(rep("Pretrain", n.alpha), rep("Individual", n.alpha), rep("Overall", n.alpha))
    )
    
    ylims2 = range(forplot2$overall)
    nudge = .1 * (ylims2[2] - ylims2[1])
    ylims[2] = ylims2[2] + nudge
    plot2 <- ggplot() +
        geom_line(aes(x=forplot2$alpha, y=forplot2$overall,
                      group = forplot2$model, color = forplot2$model)) +
         geom_text(aes(x=.2, y=err.pan["mean"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
         geom_text(aes(x=.2, y=err.ind["mean"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
         scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
         labs(x = expression(alpha), y = y.label, color = "", title=paste0("Average of ", as.character(k)," individual problems")) +
         ylim(ylims2[1], ylims2[2]) +
         theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall", "Pretrain", "Individual")) 
    
    if(plot.alphahat){
        plot1 <- plot1 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
        plot2 <- plot2 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
    }

    gridExtra::grid.arrange(plot1, plot2, ncol=2, widths=c(.8, 1))
}


#' Plot the models trained by a ptLasso object
#'
#' A plot is produced, and nothing is returned.
#'
#' @aliases plot.ptLasso 
#' @param x Fitted \code{"ptLasso"} object.
#' @param \dots Other parameters to pass to the plot function.
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' set.seed(1234)
#' out = gaussian.example.data()
#' x = out$x; y=out$y; groups = out$group
#' 
#' fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
#' plot(fit) 
#'
#' @method plot ptLasso
#' @export
plot.ptLasso = function(x, ...){
    if("nresps" %in% names(x)) x$k = x$nresps
    nm = "Group"
    if("nresps" %in% names(x)) nm = "Response"
    lo = matrix(
        c(1:x$k,                  # Overall model
          (1 + x$k) : (2*x$k),    # Pretrained models
          (1 + 2*x$k) : (3*x$k)), # Individual models
        nrow = 3,
        byrow = TRUE)
    
    graphics::par()
    graphics::layout(lo, heights=rep(1, 3))
    plot(x$fitoverall); graphics::title("Overall model", line = 2)
    for(kk in 1:(x$k - 1)) graphics::plot.new()

    line.nudge = 2
    
    for(kk in 1:x$k){
        plot(x$fitpre[[kk]]);
        if(kk == 1) {
           graphics::title(paste0("Pretrained\n", nm, " ", kk), line=line.nudge)
        } else {
           graphics::title(paste0(nm, " ", kk), line=line.nudge)
        }
        
    }

    for(kk in 1:x$k){
        plot(x$fitind[[kk]]);
        if(kk == 1) {
            graphics::title(paste0("Individual\n", nm, " ", kk), line=line.nudge)
        } else {
            graphics::title(paste0(nm, " ", kk), line=line.nudge)
        }
    }
}
