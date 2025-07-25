# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)


test.tol = 1e-2

###########################################################################
# Time series
###########################################################################
set.seed(1234)
n = 600; ntrain = 300; ncol = 50
x = matrix(rnorm(n*ncol), n, ncol)

beta1 = c(rep(0.5, 10), rep(0, ncol-10))
beta2 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(0, ncol-15))
beta3 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(.5, 5), rep(0, ncol-20))

y1 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta1)))
y2 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta2)))
y3 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta3)))
y = cbind(y1, y2, y3)

xtest = x[-(1:ntrain), ]
ytest = y[-(1:ntrain), ]

x = x[1:ntrain, ]
y = y[1:ntrain, ]

fit =  ptLasso(x, y, use.case="timeSeries", family="binomial", type.measure = "auc")
preds = predict(fit, xtest, ytest=ytest)
preds2 = predict(fit, xtest, ytest=ytest, s = "lambda.1se")

cvfit = cv.ptLasso(x, y, use.case="timeSeries", family="binomial", type.measure = "auc")
preds3 = predict(cvfit, xtest, ytest=ytest, s = "lambda.1se", alphatype = "varying")

test_that("timeseries_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 23, 24, 27, 35, 39, 43, 45, 50),
               tolerance = test.tol)
})

test_that("timeseries_suppre.common.1se", {
  expect_equal(unname(preds2$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16),
               tolerance = test.tol)
})

test_that("timeseries_suppre.common.1se.varying", {
  expect_equal(unname(preds3$suppre.common),
               c(2,  3,  4,  8,  9, 10, 13),
               tolerance = test.tol)
})

test_that("timeseries_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(18, 19, 25, 30, 31, 32, 34, 36, 37, 38, 40, 41, 42, 44, 46, 47, 49), 
               tolerance = test.tol)
})

test_that("timeseries_suppre.individual.1se", {
  expect_equal(unname(preds2$suppre.individual),
               c(11, 12, 15, 17, 18, 19, 20, 23, 24, 27, 35, 43, 44, 45), 
               tolerance = test.tol)
})

test_that("timeseries_suppre.individual.1se.varying", {
  expect_equal(unname(preds3$suppre.individual),
               c(1,  5,  6,  7, 16, 34, 39), 
               tolerance = test.tol)
})


test_that("timeseries_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35,
                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50), 
               tolerance = test.tol)
})


test_that("timeseries_supind.1se", {
  expect_equal(unname(preds2$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 23, 24, 27, 31, 34, 35, 39, 43, 44, 45), 
               tolerance = test.tol)
})


test_that("timeseries_supind.1se.varying", {
  expect_equal(unname(preds3$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 23, 24, 27, 31, 34, 35, 39, 43, 44, 45), 
               tolerance = test.tol)
})

test_that("timeseries_errpre", {
  expect_equal(unname(preds$errpre),
               c(0.8185005, 0.8193990, 0.8060476, 0.8300549), 
               tolerance = test.tol)
})


test_that("timeseries_errpre.1se", {
  expect_equal(unname(preds2$errpre),
               c(0.8125258, 0.8128610, 0.8017982, 0.8229181), 
               tolerance = test.tol)
})

test_that("timeseries_errpre.1se.varying", {
  expect_equal(unname(preds3$errpre),
               c(0.7998139, 0.8037258, 0.7834586, 0.8122575), 
               tolerance = test.tol)
})

test_that("timeseries_errind", {
  expect_equal(unname(preds$errind),
               c(0.7996871, 0.8193990, 0.7690553, 0.8106071), 
               tolerance = test.tol)
})

test_that("timeseries_errind.1se", {
  expect_equal(unname(preds2$errind),
               c(0.7890303, 0.8128610, 0.7662820, 0.7879477), 
               tolerance = test.tol)
})

test_that("timeseries_errind.1se.varying", {
  expect_equal(unname(preds3$errind),
               c(0.7859852, 0.8037258, 0.7662820, 0.7879477), 
               tolerance = test.tol)
})


set.seed(1234)
fit = ptLasso(x, y, alpha = 0.5, family = "binomial", type.measure = "deviance", use.case = "timeSeries")
preds = predict(fit, xtest, ytest=ytest) 

test_that("timeseries_dev_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
                 16, 17, 20, 23, 27, 35, 38, 39, 42, 50), 
               tolerance = test.tol)
})

test_that("timeseries_dev_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(12, 18, 19, 22, 24, 25, 31, 32, 33, 34, 37,
                 40, 43, 44, 45, 46, 47, 48, 49), 
               tolerance = test.tol)
})

test_that("timeseries_dev_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 31, 32, 33,
                 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48,
                 49, 50), 
               tolerance = test.tol)
})

test_that("timeseries_dev_errpre", {
  expect_equal(unname(preds$errpre),
               c(1.053397, 1.034874, 1.112519, 1.012797), 
               tolerance = test.tol)
})

test_that("timeseries_dev_errind", {
  expect_equal(unname(preds$errind),
               c(1.078553, 1.034874, 1.150153, 1.050633), 
               tolerance = test.tol)
})


set.seed(1234)
fit = cv.ptLasso(x, y, family = "binomial", type.measure = "auc", use.case = "timeSeries")
preds = predict(fit, xtest, ytest=ytest)

test_that("timeseries_cv_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 16, 17, 18, 20, 23, 24, 27, 32, 35, 43), 
               tolerance = test.tol)
})

test_that("timeseries_cv_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(19, 25, 30, 31, 34, 36, 37, 38, 39, 40, 41, 42, 44,
                 45, 46, 47, 49, 50), 
               tolerance = test.tol)
})


test_that("timeseries_cv_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                 46, 47, 48, 49, 50), 
               tolerance = test.tol)
})

test_that("timeseries_cv_errpre", {
  expect_equal(unname(preds$errpre),
               c(0.8207024, 0.8164883, 0.8115942, 0.8340247), 
               tolerance = test.tol)
})

test_that("timeseries_cv_errind", {
  expect_equal(unname(preds$errind),
               c(0.7986260, 0.8164883, 0.7673108, 0.8120790), 
               tolerance = test.tol)
})

######
# Now x is a list
######

set.seed(1234)
n = 600; ntrain = 300; p = 50
x = matrix(rnorm(n*p), n, p)

beta1 = c(rep(0.5, 10), rep(0, p-10))
beta2 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(0, p-15))
beta3 = beta1 + c(rep(0, 10), runif(5, min = 0, max = 0.5), rep(.5, 5), rep(0, p-20))

y1 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta1)))
y2 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta2)))
y3 = rbinom(n, 1, prob = 1/(1 + exp(-x %*% beta3)))
y = cbind(y1, y2, y3)

xtest = x[-(1:ntrain), ]
ytest = y[-(1:ntrain), ]

x = x[1:ntrain, ]
y = y[1:ntrain, ]

x = lapply(1:3, function(i) x)
xtest = lapply(1:3, function(i) xtest)

fit =  ptLasso(x, y, use.case="timeSeries", family="binomial", type.measure = "auc")
preds = predict(fit, xtest, ytest=ytest)
preds2 = predict(fit, xtest, ytest=ytest, s = "lambda.1se")

test_that("timeseries_list_cv_suppre.common", {
  expect_equal(unname(preds$suppre.common),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 20, 23, 24, 27, 35, 39, 43, 45, 50), 
               tolerance = test.tol)
})

test_that("timeseries_list_cv_suppre.individual", {
  expect_equal(unname(preds$suppre.individual),
               c(18, 19, 25, 30, 31, 32, 34, 36, 37, 38, 40, 41, 42, 44, 46, 47, 49), 
               tolerance = test.tol)
})


test_that("timeseries_list_cv_supind", {
  expect_equal(unname(preds$supind),
               c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29,
                 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                 42, 43, 44, 45, 46, 47, 48, 49, 50), 
               tolerance = test.tol)
})

test_that("timeseries_list_cv_errpre", {
  expect_equal(unname(preds$errpre),
               c(0.8185005, 0.8193990, 0.8060476, 0.8300549), 
               tolerance = test.tol)
})

test_that("timeseries_list_cv_errind", {
  expect_equal(unname(preds$errind),
               c(0.7996871, 0.8193990, 0.7690553, 0.8106071), 
               tolerance = test.tol)
})

# To test errors:
#xlist = lapply(1:3, function(i) x)
#xtestlist = lapply(1:2, function(i) xtest)
#
#fit =  ptLasso(xlist, y, use.case="timeSeries", family="binomial", type.measure = "auc")
#preds = predict(fit, xtestlist, ytest=ytest)
#
#xlist = lapply(1:2, function(i) x)
#fit =  ptLasso(xlist, y, use.case="timeSeries", family="binomial", type.measure = "auc")
