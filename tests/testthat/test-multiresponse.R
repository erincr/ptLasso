# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)


test.tol = 1e-2

###########################################################################
# Multiresponse
###########################################################################

set.seed(1234)
n = 1000; ntrain = 500;
p = 500
sigma = 2

x = matrix(rnorm(n*p), n, p)
beta1 = c(rep(1, 5), rep(0.5, 5), rep(0, p - 10))
beta2 = c(rep(1, 5), rep(0, 5), rep(0.5, 5), rep(0, p - 15))

mu = cbind(x %*% beta1, x %*% beta2)
y  = cbind(mu[, 1] + sigma * rnorm(n), 
           mu[, 2] + sigma * rnorm(n))

xtest = x[-(1:ntrain), ]
ytest = y[-(1:ntrain), ]

x = x[1:ntrain, ]
y = y[1:ntrain, ]

# Now, we can fit a ptLasso multiresponse model:
#fit = ptLassoMult(x, y, alpha = 0.5, type.measure = "mse")
fit = ptLasso(x, y, alpha = 0.5, type.measure = "mse", use.case = "multiresponse")
# plot(fit) # to see all of the cv.glmnet models trained
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 140, 257, 296, 346), 
               tolerance = test.tol)
})

test_that("multresponse_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(52, 60, 120, 153, 322, 358, 365, 400, 404, 498), 
               tolerance = test.tol)
})

test_that("multresponse_supoverall", {
  expect_equal(unname(preds$supoverall),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 52, 60, 64, 69, 99, 112, 119, 120, 125, 139, 140,
                 143, 153, 172, 179, 193, 199, 206, 223, 225, 233,
                 257, 296, 306, 313, 315, 322, 346, 357, 358, 365,
                 400, 404, 417, 418, 419, 440, 446, 457, 464, 480, 498), 
               tolerance = test.tol)
})


test_that("multresponse_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 21, 52, 60, 64, 69, 84, 89, 92, 99, 119, 120, 125,
                 130, 135, 140, 153, 171, 172, 179, 193, 199, 206,
                 215, 222, 223, 225, 233, 236, 239, 250, 252, 257,
                 265, 282, 294, 296, 301, 306, 311, 313, 315, 322,
                 342, 346, 357, 358, 365, 372, 378, 400, 402, 404,
                 412, 417, 419, 428, 440, 446, 448, 449, 455, 456,
                 464, 480, 498), 
               tolerance = test.tol)
})

test_that("multresponse_errpre", {
  expect_equal(unname(preds$errpre),
               c(9.022075, 4.511038, 4.144121, 4.877954), 
               tolerance = test.tol)
})

test_that("multresponse_errind", {
  expect_equal(unname(preds$errind),
               c(9.465296, 4.732648, 4.243210, 5.222087), 
               tolerance = test.tol)
})

test_that("multresponse_errall", {
  expect_equal(unname(preds$erroverall),
               c(9.394191, 4.697096, 4.226602, 5.167589), 
               tolerance = test.tol)
})


set.seed(1234)
fit = ptLasso(x, y, alpha = 0.5, type.measure = "mae", use.case = "multiresponse")
# plot(fit) # to see all of the cv.glmnet models trained
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_mae_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 15, 120, 140, 172, 257, 296, 346, 365, 400, 464), 
               tolerance = test.tol)
})

test_that("multresponse_mae_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(52, 60, 153, 322, 358, 404, 498), 
               tolerance = test.tol)
})

test_that("multresponse_mae_supoverall", {
  expect_equal(unname(preds$supoverall),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                 14, 15, 52, 60, 64, 69, 99, 112, 119, 120,
                 125, 139, 140, 143, 153, 172, 179, 193, 199,
                 206, 223, 225, 233, 257, 296, 306, 313, 315,
                 322, 346, 357, 358, 365, 400, 404, 417, 418,
                 419, 440, 446, 457, 464, 480, 498), 
               tolerance = test.tol)
})


test_that("multresponse_mae_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 21, 52, 60, 64, 69, 84, 89, 92, 99, 119,
                 120, 125, 130, 135, 140, 153, 171, 172, 179,
                 193, 199, 206, 215, 222, 223, 225, 233, 236,
                 239, 250, 252, 257, 265, 282, 294, 296, 301,
                 306, 311, 313, 315, 322, 342, 346, 357, 358,
                 365, 372, 378, 400, 402, 404, 412, 417, 419,
                 428, 440, 446, 448, 449, 455, 456, 464, 480, 498), 
               tolerance = test.tol)
})

test_that("multresponse_mae_errpre", {
  expect_equal(unname(preds$errpre),
               c(3.425577, 1.712789, 1.612142, 1.813435), 
               tolerance = test.tol)
})

test_that("multresponse_mae_errind", {
  expect_equal(unname(preds$errind),
               c(3.474751, 1.737375, 1.622945, 1.851805), 
               tolerance = test.tol)
})

test_that("multresponse_mae_errall", {
  expect_equal(unname(preds$erroverall),
               c(3.463764, 1.731882, 1.622082, 1.841682), 
               tolerance = test.tol)
})


set.seed(1234)
fit = cv.ptLasso(x, y, type.measure = "mae", use.case = "multiresponse")
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_cv_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 13, 14, 15, 120, 140, 172, 257, 296, 346,
                 365, 400, 464), 
               tolerance = test.tol)
})

test_that("multresponse_cv_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(60, 153, 322), 
               tolerance = test.tol)
})

test_that("multresponse_cv_supoverall", {
  expect_equal(unname(preds$supoverall),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 52, 60, 64, 69, 99, 112, 119, 120, 125, 139,
                 140, 143, 153, 172, 179, 193, 199, 206, 223, 225,
                 233, 257, 296, 306, 313, 315, 322, 346, 357, 358,
                 365, 400, 404, 417, 418, 419, 440, 446, 457, 464, 480, 498), 
               tolerance = test.tol)
})


test_that("multresponse_cv_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 21, 52, 60, 64, 69, 84, 89, 92, 99, 119,
                 120, 125, 130, 135, 140, 153, 171, 172, 179,
                 193, 199, 206, 215, 222, 223, 225, 233, 236,
                 239, 250, 252, 257, 265, 282, 294, 296, 301,
                 306, 311, 313, 315, 322, 342, 346, 357, 358,
                 365, 372, 378, 400, 402, 404, 412, 417, 419,
                 428, 440, 446, 448, 449, 455, 456, 464, 480, 498), 
               tolerance = test.tol)
})

test_that("multresponse_cv_errpre", {
  expect_equal(unname(preds$errpre),
               c(3.422234, 1.711117, 1.623132, 1.799102), 
               tolerance = test.tol)
})

test_that("multresponse_cv_errind", {
  expect_equal(unname(preds$errind),
               c(3.474751, 1.737375, 1.622945, 1.851805), 
               tolerance = test.tol)
})

test_that("multresponse_cv_errall", {
  expect_equal(unname(preds$erroverall),
               c(3.463764, 1.731882, 1.622082, 1.841682), 
               tolerance = test.tol)
})

test_that("ggplot.ptLasso.inputGroups returns gtable (multiresponse)", {
  plot_obj <- .plot_ptLasso_inputGroups(fit, y.label = "MSE")
  
  expect_true(inherits(plot_obj, "gtable"))
})

test_that("multiresponse_print_ok", {
  expect_invisible(print(fit))
})
