# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)


test.tol = 1e-2

###########################################################################
# Input groups, binomial response
###########################################################################
set.seed(1234)
n=100
k=2  # of classes
p=20

scommon=10 # # of common important  features
sindiv=c(5,1)  #of individual important features
class.sizes=2*c(30, 20)
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

cvfit=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="auc",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
pred.cv.fixed=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="fixed")

pred.test=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest,alpha=.6)

method.0=predict(cvfit$fit[[which(cvfit$alphalist == cvfit$varying.alphahat[2])]], 
                 xtest[groupstest == 2, ], groupstest=groupstest[groupstest == 2])$yhatpre
method.1=predict(cvfit, xtest[groupstest == 2,], groupstest=groupstest[groupstest == 2], alpha = cvfit$varying.alphahat[2])$yhatpre
method.2=predict(cvfit, xtest[groupstest == 2,], groupstest=groupstest[groupstest == 2], alphatype='varying')$yhatpre

fit2=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="deviance",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred2=predict(fit2,xtest,groupstest=groupstest, ytest=ytest)
cvfit2=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="deviance",foldid=NULL, nfolds=5, overall.lambda="lambda.min")


fit3=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="class",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred3=predict(fit3,xtest,groupstest=groupstest, ytest=ytest)
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
               c(0.8110790, 0.8348733, 0.8450480, 0.8857466, 0.7840000),
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_cvfit_varying_results", {
  expect_equal(unname(pred.cv$errpre),
               c(0.8726747, 0.8520437, 0.8651192, 0.9174208, 0.7866667),
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_cvfit_varying_results", {
  expect_equal(unname(pred.cv.fixed$errpre),
               c(0.8152129, 0.8316410, 0.8422359, 0.8846154, 0.7786667),
               tolerance = test.tol
  )
})

test_that("input_groups_binomial_alphahat", {
  expect_equal(cvfit$alphahat,
               0.8,
               tolerance = test.tol)
})

test_that("input_groups_binomial_alphahat_dev", {
  expect_equal(cvfit2$alphahat,
               1,
               tolerance = test.tol)
})

test_that("input_groups_binomial_alphahat_class", {
  expect_equal(cvfit3$alphahat,
               0.8,
               tolerance = test.tol)
})

test_that("input_groups_binomial_errind", {
  expect_equal(unname(pred$errind),
               c(0.8817693, 0.8473560, 0.8658938, 0.9400452, 0.7546667),
               tolerance = test.tol)
})

test_that("input_groups_binomial_errind_deviance", {
  expect_equal(as.numeric(pred2$errind),
               c(1.1306033, 1.1684537, 1.1306033, 0.9792017, 1.3577057),
               tolerance = test.tol)
})


test_that("input_groups_binomial_erroverall", {
  expect_equal(as.numeric(pred$erroverall),
               c(0.7813146, 0.8049744, 0.8209026, 0.8846154, 0.7253333),
               tolerance = test.tol)
})

test_that("input_groups_binomial_erroverall_deviance", {
  expect_equal(as.numeric(pred2$erroverall),
               c(1.277899, 1.343351, 1.277899, 1.016092, 1.670611),
               tolerance = test.tol)
})


test_that("input_groups_binomial_errpre", {
  expect_equal(unname(pred$errpre),
               c(0.8772220, 0.8650528, 0.8807300, 0.9434389, 0.7866667),
               tolerance = test.tol)
})

test_that("input_groups_binomial_errpre_deviance", {
  expect_equal(unname(pred2$errpre),
               c(1.0894876, 1.1234877, 1.0894876, 0.9534874, 1.2934880),
               tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="binomial",foldid=NULL, nfolds=3, overall.lambda = "lambda.min")
test_that("input_groups_binomial_type_measure", {
  expect_equal(check.type.default$type.measure,
               "deviance")
})

