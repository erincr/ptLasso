#'
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
#' 2+2
#'
#'
#' @import ggplot2 gridExtra
#' @method plot cv.ptLasso
#' @export
#'
#'


plot.cv.ptLasso = function(x, plot.alphahat = TRUE, ...){
    if(x$fit[[1]]$useCase == "inputGroups")  ggplot.ptLasso.inputGroups(x, plot.alphahat = plot.alphahat, y.label = "Cross validated error", ...)
    if(x$fit[[1]]$useCase == "targetGroups") ggplot.ptLasso.targetGroups(x, plot.alphahat = plot.alphahat, y.label = "Cross validated error", ...)
}

#
#' Plot the performance curve produced by predict.cv.ptLasso using a validation set, as a function of the \code{alpha} values used. 
#'
#' A plot is produced, and nothing is returned.
#'
#' @aliases plot.predict.cv.ptLasso
#' @param x \code{"predict.cv.ptLasso"} object.
#' @param plot.alphahat If \code{TRUE}, show a dashed vertical line indicating the single value of alpha that maximized overall cross-validated performance.
#' @param \dots Other graphical parameters to plot
#' @author Erin Craig and Rob Tibshirani\cr Maintainer: Erin Craig <erincr@@stanford.edu>
#' @seealso \code{ptLasso}, \code{cv.ptLasso} and \code{predict.cv.ptLasso}.
#' @keywords models regression classification
#' @examples
#' 3+3
#'
#'
#' @import ggplot2 gridExtra
#' @method plot predict.cv.ptLasso
#' @export
#'
#'
plot.predict.cv.ptLasso = function(x, plot.alphahat = TRUE, ...){
    # TODO: hook up plot.alphahat
    if(x$useCase == "inputGroups")  ggplot.ptLasso.inputGroups(x, plot.alphahat = plot.alphahat, y.label = "Validation error", ...)
    if(x$useCase == "targetGroups") ggplot.ptLasso.targetGroups(x,plot.alphahat = plot.alphahat, y.label = "Validation error", ...)
}



##################
# GGPLOT
##################
#' Plot function for target grouped data
#' @noRd
ggplot.ptLasso.targetGroups=function(x, plot.alphahat = FALSE, y.label = "Cross validated error",...){

    err.pre = x$errpre
    err.pan = x$errall
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

    k = ncol(err.pre) - 2

    ylim = range(c(err.pre[, "overallError"], err.pan["overallError"], err.ind["overallError"])) 
    nudge = .1 * (ylim[2] - ylim[1])
    ylim[2] = ylim[2] + nudge
    
    par(mfrow=c(1,2))
    par(lwd=1.5)
    par(mar=c(5,4,5,1))

    plot(err.pre[,"alpha"], err.pre[,"overallError"], ylab=y.label,xlab=expression(alpha),
         type="l",
         ylim=ylim, 
         col=3)
    lines(err.pre[,"alpha"], rep(err.pan["overallError"], n.alpha), type="l",col=2)
    lines(err.pre[,"alpha"], rep(err.ind["overallError"], n.alpha), type="l",col=1) 
    
    text(.2, err.pan + nudge/2, as.character(suppan))
    text(.2, err.ind["overallError"] + nudge/2, as.character(supind))
    axis(3,  at=err.pre[, 1], labels=as.character(suppre))#, gap.axis = 0)

    if(plot.alphahat) abline(v = x$alphahat, col = 'gray60', lty=2)
    
    title(paste0(as.character(k),"-class problem"))

    xloc = .4#2 * (min(err.pre[, 1]) + max(err.pre[, 1]))/3
    
    legend(xloc, ylim[2] - 2*nudge, 
           col=c(1:3), ncol=1, c("Individual", "Overall (grouped)", "Pretrain"),
           lwd = 3,
           xpd = NA, cex = 1,
           bty = "n",
           seg.len=1)

    group.cols = colnames(err.pre)[grepl("group", colnames(err.pre))]
    sum.of.indiv.pre = rowSums(err.pre[, group.cols])
    sum.of.indiv.ind = sum(err.ind[group.cols])
    ylim2 = range(c(sum.of.indiv.pre, sum.of.indiv.ind))
    nudge2 = .1 * (ylim2[2] - ylim2[1])
    ylim2[2] = ylim2[2] + nudge2
    
    plot(err.pre[, "alpha"], sum.of.indiv.pre, ylab=y.label,xlab=expression(alpha),
         type="l",
         ylim=ylim2, 
         col=3)
    lines(err.pre[, "alpha"], rep(sum.of.indiv.ind, nrow(err.pre)), type="l",col=1)

    if(plot.alphahat) abline(v = x$alphahat, col = 'gray60', lty=2)
    
    text(.2, sum.of.indiv.ind + nudge2/2, as.character(supind))
    axis(3,  at=err.pre[, "alpha"], labels=as.character(suppre))#, gap.axis = 0)
    title("Sum of individual\none vs. rest problems", line=2)

    legend(1.5*xloc, ylim2[2] - 2*nudge2, 
           col=c(1, 3), ncol=1, c("Individual","Pretrain"), 
           lwd = 3,
           xpd = NA, cex = 1,
           bty = "n",
           seg.len=1)
    
    invisible()
}


#' Plot function for input grouped data
#' @noRd
ggplot.ptLasso.inputGroups=function(x, plot.alphahat = FALSE, y.label = "Cross validated error",...){
    k = length(x$fitind)
    err.pre = x$errpre
    err.pan = x$errall
    err.ind = x$errind

    y.label = gsub("error", x$type.measure, y.label)

    suppre = sapply(x$fit, function(ff) length(get.pretrain.support(ff, commonOnly = FALSE)))
    supind = length(get.individual.support(x, commonOnly = FALSE))
    suppan = length(get.overall.support(x))

    n.alpha = nrow(err.pre)

    forplot = data.frame(
        "alpha"   = c(err.pre[,"alpha"], err.pre[,"alpha"], err.pre[,"alpha"]),
        "overall" = c(err.pre[, "overallError"], rep(err.ind["overallError"], n.alpha), rep(err.pan["overallError"], n.alpha)),
        "model"   = c(rep("Pretrain", n.alpha), rep("Individual", n.alpha), rep("Overall", n.alpha))
    )
    ylims = range(forplot$overall)
    nudge = .1 * (ylims[2] - ylims[1])
    ylims[2] = ylims[2] + nudge
    
    plot1 <- ggplot(forplot) +
        geom_line(aes(x=alpha, y=overall, group = model, color = model)) +
        geom_text(aes(x=.2, y=err.pan["overallError"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
        geom_text(aes(x=.2, y=err.ind["overallError"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
        scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
        labs(x = expression(alpha), y = y.label, color = "", title=paste0(as.character(k)," group problem")) +
        ylim(ylims[1], ylims[2]) +
        theme_minimal(base_size = 12) +
        scale_color_manual(values = c("#FF97C4", "#00A5CF", "#08A045")) + guides(color="none")

    forplot = data.frame(
        "alpha"   = c(err.pre[,"alpha"], err.pre[,"alpha"], err.pre[,"alpha"]),
        "overall" = c(err.pre[, "meanError"], rep(err.ind["meanError"], n.alpha), rep(err.pan["meanError"], n.alpha)),
        "model"   = c(rep("Pretrain", n.alpha), rep("Individual", n.alpha), rep("Overall", n.alpha))
    )
    ylims = range(forplot$overall)
    nudge = .1 * (ylims[2] - ylims[1])
    ylims[2] = ylims[2] + nudge
    plot2 <- ggplot(forplot) +
         geom_line(aes(x=alpha, y=overall, group = model, color = model)) +
         geom_text(aes(x=.2, y=err.pan["meanError"], label=as.character(suppan), vjust=-1), size=3, color="#666666") +
         geom_text(aes(x=.2, y=err.ind["meanError"], label=as.character(supind), vjust=-1), size=3, color="#666666") +
         scale_x_continuous(sec.axis = dup_axis(breaks = err.pre[, "alpha"][c(TRUE, FALSE)], labels = as.character(suppre)[c(TRUE, FALSE)], name = "")) + 
         labs(x = expression(alpha), y = y.label, color = "", title=paste0("Average of ", as.character(k)," individual problems")) +
         ylim(ylims[1], ylims[2]) +
         theme_minimal(base_size = 12) +
         scale_color_manual(values = c("#FF97C4", "#00A5CF", "#08A045"))


    if(plot.alphahat){
        plot1 <- plot1 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
        plot2 <- plot2 + geom_vline(aes(xintercept = x$alphahat), color = '#666666', lty=2)
    }

    gridExtra::grid.arrange(plot1, plot2, ncol=2, widths=c(.8, 1))
}

#######################################
# OLD
#######################################


#plot.ptLasso.targetGroups=function(x, plot.alphahat = FALSE, y.label = "Cross validated error",...){
#
#    err.pre = x$errpre
#    err.pan = x$errall
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
#    err.pan = x$errall
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
