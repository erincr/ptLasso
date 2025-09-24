# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)

test.tol = 1e-2

###########################################################################
# Input groups, multinomial response
###########################################################################
set.seed(1234)
n=900
k=2  # of input groups
p=50
mult.classes = 2

scommon=10 # # of common important  features
sindiv=c(10, 10)  #of individual important features
class.sizes=c(450, 450)
del=rep(2.5,k)
del2=rep(5, k)
means = rep(0, k)
sigma=20

out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma, outcome="multinomial", mult.classes=mult.classes)
x=out$x
y=out$y
groups=out$groups

out2=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
              class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
              intercepts=means, sigma=sigma, outcome="multinomial", mult.classes=mult.classes)
xtest=out2$x
groupstest=out2$groups
ytest=out2$y

fit=ptLasso(x,y,groups=groups,alpha=0.9,family="multinomial",type.measure="class",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest, type="class")
cvfit=cv.ptLasso(x,y,groups=groups,family="multinomial",type.measure="class",foldid=NULL, nfolds=3, overall.lambda="lambda.min")

test_that("input_groups_multinomial_errind", {
  expect_equal(unname(pred$errind),
               c(0.04, 0.04, 0.04, 0.05, 0.03),
               tolerance = test.tol)
})

test_that("input_groups_multinomial_erroverall_classes", {
  expect_equal(unname(pred$erroverall),
               c(0.29, 0.29, 0.29, 0.31, 0.27),
               tolerance = test.tol)
})

test_that("input_groups_multinomial_errpre", {
  expect_equal(unname(pred$errpre),
               c(0.03888889, 0.03888889, 0.03888889, 0.0511, 0.0244),
               tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="multinomial",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")

test_that("input_groups_multinomial_type_measure", {
  expect_equal(check.type.default$type.measure,
               "deviance")
})
