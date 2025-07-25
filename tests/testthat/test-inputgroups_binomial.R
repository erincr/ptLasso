# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)


test.tol = 1e-2

###########################################################################
# Input groups, binomial response
###########################################################################
set.seed(1234)
n=600
k=5  # of classes
p=140

scommon=10 # # of common important  features
sindiv=c(50,40,20,10,10)  #of individual important features
class.sizes=2*c(100,80,60,30,30)
del=rep(2.5,k)
del2=rep(5, k)
means = rep(0, k)
sigma=20

out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma, outcome="binomial")

x=out$x
y=out$y
groups=out$groups

out2=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
              class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
              intercepts=means, sigma=sigma, outcome="binomial")
xtest=out2$x
groupstest=out2$groups
ytest=out2$y

fit=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="auc",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)

cvfit=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="auc",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
pred.cv.fixed=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="fixed")

pred.test=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest,alpha=.6)

method.0=predict(cvfit$fit[[11]], xtest[groupstest == 2, ], groupstest=groupstest[groupstest == 2])$yhatpre
method.1=predict(cvfit, xtest[groupstest == 2,], groupstest=groupstest[groupstest == 2], alpha = cvfit$varying.alphahat[2])$yhatpre
method.2=predict(cvfit, xtest[groupstest == 2,], groupstest=groupstest[groupstest == 2], alphatype='varying')$yhatpre

fit2=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="deviance",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred2=predict.ptLasso(fit2,xtest,groupstest=groupstest, ytest=ytest)
cvfit2=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="deviance",foldid=NULL, nfolds=5, overall.lambda="lambda.min")


fit3=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="class",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred3=predict.ptLasso(fit3,xtest,groupstest=groupstest, ytest=ytest)
cvfit3=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="class",foldid=NULL, nfolds=5, overall.lambda="lambda.min")

test_that("input_groups_binomial_two_prediction_methods", {
  expect_equal(method.1,
               method.2,
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_two_prediction_methods", {
  expect_equal(method.0,
               method.1,
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_alpha_6", {
  expect_equal(unname(pred.test$errpre),
               c(0.7898285, 0.7710579, 0.8038164, 0.8203581, 0.8681989 ,0.8217076, 0.7144444, 0.6305804),
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_cvfit_varying_results", {
  expect_equal(unname(pred.cv$errpre),
               c(0.798, 0.775, 0.798, 0.783, 0.874, 0.824, 0.800, 0.596),
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_cvfit_varying_results", {
  expect_equal(unname(pred.cv.fixed$errpre),
               c(0.803, 0.777 ,0.803, 0.811, 0.856, 0.822 ,0.800, 0.596),
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_alphahat", {
  expect_equal(cvfit$alphahat,
               1,
               tolerance = test.tol)
})

test_that("input_groups_binomial_alphahat_dev", {
  expect_equal(cvfit2$alphahat,
               0.9,
               tolerance = test.tol)
})

test_that("input_groups_binomial_alphahat_class", {
  expect_equal(cvfit3$alphahat,
               0.8,
               tolerance = test.tol)
})

test_that("input_groups_binomial_errind", {
  expect_equal(unname(pred$errind),
               c(0.793, 0.797, 0.811, 0.825, 0.850, 0.779, 0.812, 0.720),
               tolerance = test.tol)
})

test_that("input_groups_binomial_errind_deviance", {
  expect_equal(as.numeric(pred2$errind),
               c(1.175, 1.223, 1.175, 1.169, 1.036, 1.176, 1.371, 1.364),
               tolerance = test.tol)
})


test_that("input_groups_binomial_erroverall", {
  expect_equal(as.numeric(pred$erroverall),
               c(0.6852769, 0.6801096, 0.6916161, 0.7209207,
                 0.6976235, 0.6707589, 0.5311111, 0.7801339),
               tolerance = test.tol)
})

test_that("input_groups_binomial_erroverall_deviance", {
  expect_equal(as.numeric(pred2$erroverall),
               c(1.359, 1.357, 1.359, 1.345, 1.394, 1.339, 1.434, 1.275),
               tolerance = test.tol)
})


test_that("input_groups_binomial_errpre", {
  expect_equal(unname(pred$errpre),
               c(0.799, 0.801, 0.817, 0.829, 0.866, 0.792, 0.794, 0.723),
               tolerance = test.tol)
})

test_that("input_groups_binomial_errpre_deviance", {
  expect_equal(unname(pred2$errpre),
               c(1.173, 1.219, 1.173, 1.176, 1.051, 1.134, 1.387, 1.345),
               tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="binomial",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
test_that("input_groups_binomial_type_measure", {
  expect_equal(check.type.default$type.measure,
               "deviance")
})

