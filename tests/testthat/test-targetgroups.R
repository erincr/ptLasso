# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)


test.tol = 1e-2

###########################################################################
# Target groups
###########################################################################
set.seed(2345)
n=300
p=140

out=makedata.targetgroups(n=n, p=p, scommon = 10, sindiv = rep(10, 3), class.sizes=c(100, 100, 100), shift.common=rep(1, 3), shift.indiv=rep(1, 3))
x=out$x
y=out$y
groups=y 

out2=makedata.targetgroups(n=n, p=p, scommon = 10, sindiv = rep(10, 3),  class.sizes=c(100, 100, 100), shift.common=rep(1, 3), shift.indiv=rep(1, 3))
xtest=out2$x
ytest=out2$y
groupstest=ytest

fit=ptLasso(x,y,alpha=0.222,family="multinomial",use.case="targetGroups", type.measure="class",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred=predict(fit,xtest,ytest=ytest)

fit2=ptLasso(x,y,groups=groups,alpha=0.222,family="multinomial",use.case="targetGroups", type.measure="deviance",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred2=predict.ptLasso(fit2,xtest, ytest=ytest)

cvfit = cv.ptLasso(x,y,groups=groups,family="multinomial",type.measure="class", use.case="targetGroups", foldid=NULL, nfolds=3, overall.lambda="lambda.min")
cvfit2 = cv.ptLasso(x,y,groups=groups,family="multinomial",type.measure="deviance", use.case="targetGroups", foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred2=predict(cvfit2,xtest, ytest=ytest)
pred3=predict(cvfit2,xtest, ytest=ytest, alphatype = "varying")

test_that("target_groups_print_ok", {
  expect_invisible(print(cvfit))
})

test_that("target_groups_cvplot_ok", {
  plot_obj <- plot(cvfit)
  
  expect_true(inherits(plot_obj, "gtable"))
})

test_that("target_groups_misclassification_alphahat", {
  expect_equal(cvfit$alphahat,
               1,
               tolerance = test.tol)
})

test_that("target_groups_deviance_alphahat", {
  expect_equal(cvfit2$alphahat,
               0.5,
               tolerance = test.tol)
})

test_that("target_groups_misclassification_errind", {
  expect_equal(unname(pred$errind),
               c(0.033, 0.056, 0.060, 0.057, 0.050),
               tolerance = test.tol)
})

test_that("target_groups_deviance_errind", {
  expect_equal(unname(pred2$errind),
               c(0.235, 0.252, 0.256, 0.274, 0.225),
               tolerance = test.tol)
})


test_that("target_groups_misclassification_errpre", {
  expect_equal(as.numeric(pred$errpre),
               c(0.050, 0.058, 0.060, 0.053, 0.060), 
               tolerance = test.tol)
})

test_that("target_groups_deviance_errpre", {
  expect_equal(as.numeric(pred2$errpre),
               c(0.305, 0.260, 0.279, 0.288, 0.214), 
               tolerance = test.tol)
})

test_that("target_groups_misclassification_erroverall", {
  expect_equal(as.numeric(pred$erroverall),
               c(0.033,   NA,   NA,   NA,   NA), 
               tolerance = test.tol)
})


test_that("target_groups_deviance_erroverall", {
  expect_equal(as.numeric(pred2$erroverall),
               c(0.254,   NA,   NA,   NA,   NA), 
               tolerance = test.tol)
})


test_that("target_groups_misclassification_varying_alpha", {
  expect_equal(unname(pred3$errpre),
               c(0.334, 0.276, 0.318, 0.288, 0.222), 
               tolerance = test.tol)
})


test_that("target_groups_misclassification_varying_alpha_errind_same", {
  expect_equal(unname(pred2$errind),
               unname(pred3$errind), 
               tolerance = test.tol)
})