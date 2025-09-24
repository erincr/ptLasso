# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)


test.tol = 1e-2

###########################################################################
# Multiresponse
###########################################################################

set.seed(1234)
n = 500; ntrain = 250;
p = 100
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
               c(1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 63),
               tolerance = test.tol)
})

test_that("multresponse_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(7, 32, 33, 34, 39, 75, 90, 91, 95, 98), 
               tolerance = test.tol)
})

test_that("multresponse_supoverall", {
  expect_equal(unname(preds$supoverall),
               c(1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 
                 13, 14, 15, 22, 32, 33, 34, 37, 38, 39, 41, 
                 50, 51, 59, 63, 75, 82, 85, 90, 91, 93, 94, 95, 98), 
               tolerance = test.tol)
})


test_that("multresponse_supind", {
  expect_equal(unname(preds$supind),
               c(1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 
                 13, 14, 15, 22, 30, 32, 33, 34, 37, 38, 39, 41, 
                 47, 50, 51, 54, 59, 63, 70, 75, 78, 82, 85, 90, 91, 93, 95, 98), 
               tolerance = test.tol)
})

test_that("multresponse_errpre", {
  expect_equal(unname(preds$errpre),
               c(8.838686, 4.419343, 3.790327, 5.048359), 
               tolerance = test.tol)
})

test_that("multresponse_errind", {
  expect_equal(unname(preds$errind),
               c(9.063094, 4.531547, 4.070806, 4.992287), 
               tolerance = test.tol)
})

test_that("multresponse_errall", {
  expect_equal(unname(preds$erroverall),
               c(9.197289, 4.598645, 4.005108, 5.192181), 
               tolerance = test.tol)
})


set.seed(1234)
fit = ptLasso(x, y, alpha = 0.5, nfolds = 3, type.measure = "mae", use.case = "multiresponse")
# plot(fit) # to see all of the cv.glmnet models trained
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_mae_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c( 1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 63), 
               tolerance = test.tol)
})

test_that("multresponse_mae_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(7), 
               tolerance = test.tol)
})

test_that("multresponse_mae_supoverall", {
  expect_equal(unname(preds$supoverall),
               c( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 22, 
                  32, 33, 34, 37, 38, 39, 41, 50, 51, 63, 75, 82, 85, 90, 93, 98), 
               tolerance = test.tol)
})


test_that("multresponse_mae_supind", {
  expect_equal(unname(preds$supind),
               c( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 
                  32, 33, 34, 38, 39, 50, 63, 75, 82, 90, 91, 98), 
               tolerance = test.tol)
})

test_that("multresponse_mae_errpre", {
  expect_equal(unname(preds$errpre),
               c(3.385598, 1.692799, 1.569537, 1.816061), 
               tolerance = test.tol)
})

test_that("multresponse_mae_errind", {
  expect_equal(unname(preds$errind),
               c(3.489664, 1.744832, 1.631055, 1.858609), 
               tolerance = test.tol)
})

test_that("multresponse_mae_errall", {
  expect_equal(unname(preds$erroverall),
               c(3.449901, 1.724950, 1.601631, 1.848270), 
               tolerance = test.tol)
})


set.seed(1234)
fit = cv.ptLasso(x, y, type.measure = "mae", nfolds = 3, use.case = "multiresponse")
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_cv_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c( 1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 63), 
               tolerance = test.tol)
})

test_that("multresponse_cv_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(7, 32, 33, 34, 38, 39, 50, 75, 82, 90, 91, 98), 
               tolerance = test.tol)
})

test_that("multresponse_cv_supoverall", {
  expect_equal(unname(preds$supoverall),
               c(1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 
                 22, 32, 33, 34, 37, 38, 39, 41, 50, 51, 63, 75, 82, 85, 90, 93, 98), 
               tolerance = test.tol)
})


test_that("multresponse_cv_supind", {
  expect_equal(unname(preds$supind),
               c(1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 
                 32, 33, 34, 38, 39, 50, 63, 75, 82, 90, 91, 98), 
               tolerance = test.tol)
})

test_that("multresponse_cv_errpre", {
  expect_equal(unname(preds$errpre),
               c(3.489664, 1.744832, 1.631055, 1.858609), 
               tolerance = test.tol)
})

test_that("multresponse_cv_errind", {
  expect_equal(unname(preds$errind),
               c(3.489664, 1.744832, 1.631055, 1.858609), 
               tolerance = test.tol)
})

test_that("multresponse_cv_errall", {
  expect_equal(unname(preds$erroverall),
               c(3.449901, 1.724950, 1.601631, 1.848270), 
               tolerance = test.tol)
})

test_that("ggplot.ptLasso.inputGroups returns gtable (multiresponse)", {
  plot_obj <- .plot_ptLasso_inputGroups(fit, y.label = "MSE")
  
  expect_true(inherits(plot_obj, "gtable"))
})

test_that("multiresponse_print_ok", {
  expect_invisible(print(fit))
})
