# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")


library(ptLasso)

test.tol = 1e-2

###########################################################################
# Cox
###########################################################################
set.seed(2345)
n=300
k=5  # of classes
p=140

scommon=10 # # of common important  features
sindiv=c(50,40,20,10,10)  #of individual important features
class.sizes=c(100,80,60,30,30)
del=rep(5,k)
del2=rep(4, k)
means = sample(1:k)
sigma=20

out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma, outcome="gaussian")

x=out$x
y=out$y
y=y-min(y)+.1
status=sample(c(0,1),size=n,rep=T)

y=cbind(y,status)
colnames(y)=c("time","status")

groups=out$groups

out2=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
              class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
              intercepts=means, sigma=sigma, outcome="gaussian")
xtest=out2$x
groupstest=out2$groups
ytest=out2$y
ytest=ytest-min(ytest)+.1
statustest=sample(c(0,1),size=n,rep=T)

ytest=cbind(ytest,statustest)
colnames(ytest)=c("time","status")

fit=ptLasso(x,y,groups=groups,alpha=0.1,family="cox",type.measure="C",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)

cvfit = cv.ptLasso(x,y,groups=groups,family="cox",type.measure="C",foldid=NULL, nfolds=5, overall.lambda="lambda.min")

test_that("input_groups_cox_alphahat", {
  expect_equal(cvfit$alphahat,
               0.5,
               tolerance = test.tol)
})

test_that("input_groups_cox_errind", {
  expect_equal(unname(pred$errind),
               c( 0.544, 0.564, 0.564, 0.612, 0.510, 0.543, 0.627, 0.528),
               tolerance = test.tol)
})


test_that("input_groups_cox_erroverall_classes", {
  expect_equal(unname(pred$erroverall),
               c(0.546, 0.523, 0.537, 0.549, 0.491, 0.652, 0.389, 0.535),
               tolerance = test.tol)
})

test_that("input_groups_cox_errpre", {
  expect_equal(unname(pred$errpre),
               c(0.558, 0.555, 0.560, 0.591, 0.492, 0.623, 0.516, 0.550),
               tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="cox",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
test_that("input_groups_cox_type_measure", {
  expect_equal(check.type.default$type.measure,
               "deviance")
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="cox",foldid=NULL, nfolds=5,type.measure="mse", overall.lambda = "lambda.min")
test_that("input_groups_cox_wrong_type_measure", {
  expect_equal(check.type.default$type.measure,
               "deviance")
})

test_that("input_groups_print_ok", {
  expect_invisible(print(cvfit))
})


test_that("ggplot.ptLasso.inputGroups returns gtable (cox)", {
  plot_obj <- .plot_ptLasso_inputGroups(cvfit, y.label = "MSE")
  
  expect_true(inherits(plot_obj, "gtable"))
})