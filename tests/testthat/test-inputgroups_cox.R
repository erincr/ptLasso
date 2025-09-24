# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")


library(ptLasso)

test.tol = 1e-2

###########################################################################
# Cox
###########################################################################
set.seed(2345)
n=100
k=2  # of classes
p=10

scommon=3 # # of common important  features
sindiv=c(3, 2)  #of individual important features
class.sizes=c(60, 40)
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

fit=ptLasso(x,y,groups=groups,alpha=0.1,family="cox",type.measure="C",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)

cvfit = cv.ptLasso(x,y,groups=groups,family="cox",type.measure="C",foldid=NULL, nfolds=3, overall.lambda="lambda.min")

test_that("input_groups_cox_alphahat", {
  expect_equal(cvfit$alphahat,
               0.5,
               tolerance = test.tol)
})

test_that("input_groups_cox_errind", {
  expect_equal(unname(pred$errind),
               c(0.5223624, 0.5602894, 0.5482315, 0.5000000, 0.6205788),
               tolerance = test.tol)
})


test_that("input_groups_cox_erroverall_classes", {
  expect_equal(unname(pred$erroverall),
               c(0.6487003, 0.6755914, 0.6788769, 0.6920188, 0.6591640),
               tolerance = test.tol)
})

test_that("input_groups_cox_errpre", {
  expect_equal(unname(pred$errpre),
               c(0.6467890, 0.6707682, 0.6750183, 0.6920188, 0.6495177),
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


# test_that("ggplot.ptLasso.inputGroups returns gtable (cox)", {
#   plot_obj <- .plot_ptLasso_inputGroups(cvfit, y.label = "MSE")
#   
#   expect_true(inherits(plot_obj, "gtable"))
# })
