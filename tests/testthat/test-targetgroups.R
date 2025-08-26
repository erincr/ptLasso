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
p=10

out=makedata.targetgroups(n=n, p=p, scommon = 2, sindiv = rep(2, 3), 
                          class.sizes=c(100, 100, 100), 
                          shift.common=rep(1, 3), 
                          shift.indiv=rep(1, 3))
x=out$x
y=out$y
groups=y 

out2=makedata.targetgroups(n=n, p=p, scommon = 2, sindiv = rep(2, 3), 
                           class.sizes=c(100, 100, 100), 
                           shift.common=rep(1, 3),
                           shift.indiv=rep(1, 3))
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
               0.6,
               tolerance = test.tol)
})

test_that("target_groups_deviance_alphahat", {
  expect_equal(cvfit2$alphahat,
               1,
               tolerance = test.tol)
})

test_that("target_groups_misclassification_errind", {
  expect_equal(unname(pred3$errpre),
               c(1.5557892, 0.8660415, 0.8708201, 0.9883099, 0.7389946), 
               tolerance = test.tol)
})

test_that("target_groups_deviance_errind", {
  expect_equal(unname(pred2$errind),
               c(1.5006658, 0.8597625, 0.8531700, 0.9874576, 0.7386598), 
               tolerance = test.tol)
})


test_that("target_groups_misclassification_errpre", {
  expect_equal(unname(pred$errpre),
               c(0.2933333, 0.2100000, 0.2133333, 0.2366667, 0.1800000), 
               tolerance = test.tol)
})

test_that("target_groups_deviance_errpre", {
  expect_equal(as.numeric(pred2$errpre),
               c(1.5006658, 0.8597625, 0.8531700, 0.9874576, 0.7386598), 
               tolerance = test.tol)
})

test_that("target_groups_misclassification_erroverall", {
  expect_equal(as.numeric(pred$erroverall),
               c(0.3066667,   NA,   NA,   NA,   NA), 
               tolerance = test.tol)
})


test_that("target_groups_deviance_erroverall", {
  expect_equal(as.numeric(pred2$erroverall),
               c(1.367519,   NA,   NA,   NA,   NA), 
               tolerance = test.tol)
})


test_that("target_groups_misclassification_varying_alpha", {
  expect_equal(unname(pred3$errpre),
               c(1.5557892, 0.8660415, 0.8708201, 0.9883099, 0.7389946), 
               tolerance = test.tol)
})


test_that("target_groups_misclassification_varying_alpha_errind_same", {
  expect_equal(unname(pred2$errind),
               unname(pred3$errind), 
               tolerance = test.tol)
})