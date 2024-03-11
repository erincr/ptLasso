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
    if(x$call$use.case == "inputGroups")  ggplot.ptLasso.inputGroups(x,  plot.alphahat = plot.alphahat, y.label = yaxis.name(x$call$type.measure), ...)
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
    
    plot1 <- ggplot(forplot) +
        geom_line(aes(x=alpha, y=overall, group = model, color = model)) +
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
    forplot <- data.frame(
        "alpha"          = c(err.pre[,"alpha"], err.pre[,"alpha"]),
        "individualsum"  = c(sum.of.indiv.pre, rep(sum.of.indiv.ind, n.alpha)),
        "model"          = factor(c(rep("Pretrain", n.alpha), rep("Individual", n.alpha)), levels = c("Pretrain", "Individual", "Overall\n(grouped)"))
    )
    ylims =range(forplot$individualsum)
    nudge = .1 * (ylims[2] - ylims[1])
    ylims[2] = ylims[2] + nudge

    plot2 <- ggplot(forplot) +
        geom_line(aes(x=alpha, y=individualsum, group = model, color = model)) +
        geom_text(aes(x=.2, y=sum.of.indiv.ind, label=as.character(supind), vjust=-1), size=3, color="#666666") +
        scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
        labs(x = expression(alpha), y = y.label, color = "", subtitle="Sum of individual one vs. rest problems") +
        ylim(ylims[1], ylims[2]) +
        theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall\n(grouped)", "Pretrain", "Individual"), drop = FALSE) 

    if(plot.alphahat) {
        plot1 = plot1 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
        plot2 = plot2 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
    }
    
    gridExtra::grid.arrange(plot1, plot2, ncol=2, widths=c(.75, 1))
    #print(plot1 + plot2 + plot_layout(widths = c(9, 9)))

}


#' Plot function for input grouped data
#' @noRd
ggplot.ptLasso.inputGroups=function(x, y.label, plot.alphahat = FALSE,...){
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
    
    plot1 <- ggplot(forplot) +
        geom_line(aes(x=alpha, y=overall, group = model, color = model)) +
        geom_text(aes(x=.2, y=err.pan["overall"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
        geom_text(aes(x=.2, y=err.ind["overall"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
        scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
        labs(x = expression(alpha), y = y.label, color = "", title=paste0(as.character(k)," group problem")) +
        ylim(ylims[1], ylims[2]) +
        theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall", "Pretrain", "Individual"))

    if(y.label == "Mean squared error") {
        # Average of group-specific problems is the same as the overall model
        print(plot1)
        return()
    }
    plot1 = plot1 + guides(color="none")

    forplot = data.frame(
        "alpha"   = c(err.pre[,"alpha"], err.pre[,"alpha"], err.pre[,"alpha"]),
        "overall" = c(err.pre[, "mean"], rep(err.ind["mean"], n.alpha), rep(err.pan["mean"], n.alpha)),
        "model"   = c(rep("Pretrain", n.alpha), rep("Individual", n.alpha), rep("Overall", n.alpha))
    )
    ylims = range(forplot$overall)
    nudge = .1 * (ylims[2] - ylims[1])
    ylims[2] = ylims[2] + nudge
    plot2 <- ggplot(forplot) +
         geom_line(aes(x=alpha, y=overall, group = model, color = model)) +
         geom_text(aes(x=.2, y=err.pan["mean"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
         geom_text(aes(x=.2, y=err.ind["mean"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
         scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
         labs(x = expression(alpha), y = y.label, color = "", title=paste0("Average of ", as.character(k)," individual problems")) +
         ylim(ylims[1], ylims[2]) +
         theme_minimal(base_size = 12) +
        scale_color_manual(values = c(overall.color, pretrain.color, individual.color), breaks = c("Overall", "Pretrain", "Individual")) 

    if(plot.alphahat){
        plot1 <- plot1 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
        plot2 <- plot2 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
    }

    gridExtra::grid.arrange(plot1, plot2, ncol=2, widths=c(.8, 1))
    #print(plot1 + plot2 + plot_layout(widths = c(9, 9)))
}


#' Plot the models trained by a ptLasso object
#'
#' A plot is produced, and nothing is returned.
#'
#' @aliases plot.ptLasso 
#' @param fit Fitted \code{"ptLasso"} object.
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
plot.ptLasso = function(fit){
    lo = matrix(
        c(rep(1, fit$k),                      # Title: "Overall model"
          2:(2 + fit$k - 1),                  # Overall model
          rep(2 + fit$k, fit$k),              # Title: "Pretrained models"
          (3 + fit$k) : (3 + 2*fit$k - 1),    # Pretrained models
          rep(3 + 2*fit$k, fit$k),            # Title: "Individual models"
          (4 + 2*fit$k) : (4 + 3*fit$k - 1)), # Individual models
        nrow = 6,
        byrow = TRUE)
    
    par(mar=c(1.8,3,1,2))
    layout(lo, heights=rep(c(1,3), 3))
    plot.new(); text(0.5,0.5,"Overall model", font=2, cex=1.5);
    plot(fit$fitoverall); for(kk in 1:(fit$k - 1)) plot.new()

    line.nudge = -1
    if(inherits(fit$fitoverall, "cv.sparsenet")) line.nudge = -1.5
    
    plot.new(); text(0.5,0.5,"Pretrained models", font=2, cex=1.5);
    for(kk in 1:fit$k){
        plot(fit$fitpre[[kk]]);
        title(paste0("Group ", kk), line=line.nudge)
    }

    plot.new(); text(0.5,0.5,"Individual models", font=2, cex=1.5);
    for(kk in 1:fit$k){
        plot(fit$fitind[[kk]]);
        title(paste0("Group ", kk), line=line.nudge)
    }
    
}


#######################################
# OLD
#######################################


#plot.ptLasso.targetGroups=function(x, plot.alphahat = FALSE, y.label = "Cross validated error",...){
#
#    err.pre = x$errpre
#    err.pan = x$erroverall
#    err.ind = x$errind
#    n.alpha = nrow(err.pre)
#
#    if(is.null(x$suppre)){
#        suppre = sapply(x$fit, function(ff) length(get.pretrain.support(ff, commonOnly = FALSE)))
#        supind = length(get.individual.support(x, commonOnly = FALSE))
#        suppan = length(get.overall.support(x))
#    } else {
#        suppre = sapply(x$suppre, function(ff) length(ff))
#        supind = length(x$supind)
#        suppan = length(x$supall)
#    }
#
#    k = ncol(err.pre) - 2
#
#    ylim = range(c(err.pre[, "overallError"], err.pan["overallError"], err.ind["overallError"])) 
#    nudge = .1 * (ylim[2] - ylim[1])
#    ylim[2] = ylim[2] + nudge
#    
#    par(mfrow=c(1,2))
#    par(lwd=1.5)
#    par(mar=c(5,4,5,1))
#
#    plot(err.pre[,"alpha"], err.pre[,"overallError"], ylab=y.label,xlab=expression(alpha),
#         type="l",
#         ylim=ylim, 
#         col=3)
#    lines(err.pre[,"alpha"], rep(err.pan["overallError"], n.alpha), type="l",col=2)
#    lines(err.pre[,"alpha"], rep(err.ind["overallError"], n.alpha), type="l",col=1) 
#    
#    text(.2, err.pan + nudge/2, as.character(suppan))
#    text(.2, err.ind["overallError"] + nudge/2, as.character(supind))
#    axis(3,  at=err.pre[, 1], labels=as.character(suppre))#, gap.axis = 0)
#
#    if(plot.alphahat) abline(v = x$alphahat, col = 'gray60', lty=2)
#    
#    title(paste0(as.character(k),"-class problem"))
#
#    xloc = .4#2 * (min(err.pre[, 1]) + max(err.pre[, 1]))/3
#    
#    legend(xloc, ylim[2] - 2*nudge, 
#           col=c(1:3), ncol=1, c("Individual", "Overall (grouped)", "Pretrain"),
#           lwd = 3,
#           xpd = NA, cex = 1,
#           bty = "n",
#           seg.len=1)
#
#    group.cols = colnames(err.pre)[grepl("group", colnames(err.pre))]
#    sum.of.indiv.pre = rowSums(err.pre[, group.cols])
#    sum.of.indiv.ind = sum(err.ind[group.cols])
#    ylim2 = range(c(sum.of.indiv.pre, sum.of.indiv.ind))
#    nudge2 = .1 * (ylim2[2] - ylim2[1])
#    ylim2[2] = ylim2[2] + nudge2
#    
#    plot(err.pre[, "alpha"], sum.of.indiv.pre, ylab=y.label,xlab=expression(alpha),
#         type="l",
#         ylim=ylim2, 
#         col=3)
#    lines(err.pre[, "alpha"], rep(sum.of.indiv.ind, nrow(err.pre)), type="l",col=1)
#
#    if(plot.alphahat) abline(v = x$alphahat, col = 'gray60', lty=2)
#    
#    text(.2, sum.of.indiv.ind + nudge2/2, as.character(supind))
#    axis(3,  at=err.pre[, "alpha"], labels=as.character(suppre))#, gap.axis = 0)
#    title("Sum of individual\none vs. rest problems", line=2)
#
#    legend(1.5*xloc, ylim2[2] - 2*nudge2, 
#           col=c(1, 3), ncol=1, c("Individual","Pretrain"), 
#           lwd = 3,
#           xpd = NA, cex = 1,
#           bty = "n",
#           seg.len=1)
#    
#    invisible()
#}


#plot.ptLasso.inputGroups=function(x, plot.alphahat = FALSE, y.label = "Cross validated error",...){
#                                        # par(mfrow=c(1,3))
#    
#    k = length(x$fitind)
#    err.pre = x$errpre
#    err.pan = x$erroverall
#    err.ind = x$errind
#
#    y.label = gsub("error", x$type.measure, y.label)
#
#    suppre = sapply(x$fit, function(ff) length(get.pretrain.support(ff, commonOnly = FALSE)))
#    supind = length(get.individual.support(x, commonOnly = FALSE))
#    suppan = length(get.overall.support(x))
#
#    n.alpha = nrow(err.pre)
#    
#    par(mfrow=c(1,2))
#    par(lwd=1.5)
#    par(mar=c(5,4,5,1))
#
#    ylim = range(c(err.pre[, "overallError"], err.pan["overallError"], err.ind["overallError"])) 
#    nudge = .1 * (ylim[2] - ylim[1])
#    ylim[2] = ylim[2] + nudge
#    plot(err.pre[,"alpha"], err.pre[,"overallError"], ylab=y.label,xlab=expression(alpha),
#         type="l",
#         ylim=ylim, 
#         col=3)
#    lines(err.pre[,"alpha"], rep(err.pan["overallError"], n.alpha), type="l",col=2)
#    lines(err.pre[,"alpha"], rep(err.ind["overallError"], n.alpha), type="l",col=1) 
#    
#    text(.2, err.pan["overallError"] + nudge/2, as.character(suppan))
#    text(.2, err.ind["overallError"] + nudge/2, as.character(supind))
#    axis(3,  at=err.pre[, 1], labels=as.character(suppre))#, gap.axis = 0)
#
#    if(plot.alphahat) abline(v = x$alphahat, col = 'gray60', lty=2)
#    
#    title(paste0(as.character(k),"-group problem"))
#
#    xloc = 0.3
#    legend(xloc, ylim[2] - 2*nudge, 
#           col=c(1:3), ncol=1, c("Individual", "Overall (grouped)", "Pretrain"),
#           lwd = 3,
#           xpd = NA, cex = 1,
#           bty = "n",
#           seg.len=1)
#
#    ylim = range(c(err.pre[, "meanError"], err.ind["meanError"]))
#    nudge = .1 * (ylim[2] - ylim[1])
#    ylim[2] = ylim[2] + nudge
#    plot(err.pre[,"alpha"], err.pre[,"meanError"], ylab=y.label,xlab=expression(alpha),
#         type="l",
#         ylim=ylim, 
#         col=3)
#    lines(err.pre[,"alpha"], rep(err.ind["meanError"], n.alpha), type="l",col=1) 
#    
#    text(.2, err.ind["meanError"] + nudge/2, as.character(supind))
#    axis(3,  at=err.pre[, 1], labels=as.character(suppre))#, gap.axis = 0)
#
#    if(plot.alphahat) abline(v = x$alphahat, col = 'gray60', lty=2)
#    
#    title(paste0("Average of ", as.character(k)," individual problems"))
#
#     legend(1.5*xloc, ylim[2] - 2*nudge, 
#           col=c(1, 3), ncol=1, c("Individual","Pretrain"), 
#           lwd = 3,
#           xpd = NA, cex = 1,
#           bty = "n",
#           seg.len=1)
#}
