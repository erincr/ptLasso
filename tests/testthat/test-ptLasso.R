# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

require(glmnet)
require(grf)
require(pROC)

#path = "~/Dropbox/pretrain/erin/workspace/ptLasso/R/"
#source(paste0(path, "ptlasso.R"))
#source(paste0(path, "cv.ptLasso.R"))
#source(paste0(path, "predict.ptLasso.R"))
#source(paste0(path, "helper.R"))

#path2 = "~/Dropbox/pretrain/erin/workspace/"
#source(paste0(path2, "morefuns.R"))

test.tol = 1e-2

##########################################################################################
# Input groups, Gaussian response
##########################################################################################

set.seed(1234)
n=300
k=5  # of classes
p=140

scommon=10 # # of common important  features
sindiv=c(50,40,20,10,10)  #of individual important features
class.sizes=c(100,80,60,30,30)
del=rep(2.5,k)
del2=rep(5, k)
means = sample(1:k)
sigma=20

out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma)

x=out$x
y=out$y
groups=out$groups

nfolds=5
foldid = rep(1, nrow(x))  
for(kk in 1:k) foldid[groups == kk] = sample(1:nfolds, class.sizes[kk], replace=TRUE)

out2=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma)
xtest=out2$x
groupstest=out2$groups
ytest=out2$y

fit=ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",type.measure="mse",foldid=foldid, overall.lambda = "lambda.min")
pred=predict.ptLasso(fit,xtest,groupstest=groupstest, ytest=ytest)
set.seed(1234)
cvfit=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")

pred.cv.gp2=predict(cvfit,xtest[groupstest == 2, ],groupstest=groupstest[groupstest == 2], ytest=ytest[groupstest == 2], alphatype="varying")

fit2=ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",type.measure="mae",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
pred2=predict.ptLasso(fit2,xtest,groupstest=groupstest, ytest=ytest)
set.seed(1234)
cvfit2=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mae",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")

fit3=ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",foldid=foldid, overall.lambda = "lambda.min") # should be mse
pred3=predict(fit3,xtest,groupstest=groupstest, ytest=ytest)

test_that("input_groups_gaussian_varying_alpha", {
    expect_equal(pred.cv$alpha,
                 c(0.2, 1.0, 0.8, 0.4, 0.8),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_varying_alpha_group2", {
    expect_equal(pred.cv.gp2$yhatpre,
                 pred.cv$yhatpre[groupstest == 2],
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_default_typemeasure", {
    expect_equal(pred3$errall,
                 pred$errall,
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errind", {
    expect_equal(unname(pred$errind),
                 c(1220.0722, 1108.2213, 1220.0722, 1397.8612, 1344.5456, 1157.0300,  755.8094, 885.8602),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errind_mae", {
    expect_equal(unname(pred2$errind),
                 c(26.81726, 25.54627, 26.81726, 28.35398, 29.66716, 24.83674, 20.92532, 23.94817),
                 tolerance = test.tol)
})


test_that("input_groups_gaussian_errall_classes", {
    expect_equal(unname(pred$errall),
                 c(1557.3850, 1385.5203, 1557.3850, 1808.8799, 1821.3583, 1389.9090,  780.7341, 1126.7373),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errall_classes_mae", {
    expect_equal(unname(pred2$errall),
                 c(30.07361, 28.13251, 30.07361, 33.17894, 33.70687, 26.47559, 21.75913, 25.54100),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errpre", {
    expect_equal(unname(pred$errpre),
                 c(1231.9946, 1113.9405, 1231.9946, 1478.0702, 1292.9492, 1146.4981,  759.9986, 892.1862),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errpre_mae", {
    expect_equal(unname(pred2$errpre),
                 c(27.17871, 25.87724, 27.17871, 28.97823, 29.85613, 25.02517, 21.50050, 24.02635),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_cvfit", {
    expect_equal(unname(cvfit$errpre[, "mean"]),
                 c(1078.2883, 1113.6082, 1045.0445, 1048.0911, 1012.7833,  947.6540,  940.9555,
                   911.3491,  961.1960,  944.3964,  925.0839),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_cvfit_mae", {
    expect_equal(unname(cvfit2$errpre[, "mean"]),
                 c(26.04082, 26.67820, 25.52361, 25.58128, 25.12847, 24.36098, 24.18239, 23.69399,
                   24.15136, 24.13376, 23.67217),
                 tolerance = test.tol)
})

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

# Erin: check this works if indexed at 0 - just error and message
# CV alpha: choose best performance e.g. just for one group
# check size of glmnet objects
fit=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="auc",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)

cvfit=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="auc",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
pred.cv.fixed=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="fixed")


fit2=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="deviance",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred2=predict.ptLasso(fit2,xtest,groupstest=groupstest, ytest=ytest)
cvfit2=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="deviance",foldid=NULL, nfolds=5, overall.lambda="lambda.min")


fit3=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="class",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred3=predict.ptLasso(fit3,xtest,groupstest=groupstest, ytest=ytest)
cvfit3=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="class",foldid=NULL, nfolds=5, overall.lambda="lambda.min")

test_that("input_groups_binomial_cvfit_varying_results", {
    expect_equal(unname(pred.cv$errpre),
                 c(0.8152516, 0.7919240, 0.8153424, 0.8314066,
                   0.8616323, 0.8178013, 0.7233333, 0.7254464),
                 tolerance = test.tol
                 )
})

test_that("input_groups_binomial_cvfit_varying_results", {
    expect_equal(unname(pred.cv.fixed$errpre),
                 c(0.7982510, 0.7677278, 0.7992814, 0.8243478,
                   0.8616323, 0.7946429, 0.8044444, 0.5535714),
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
                 0.9,
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errind", {
    expect_equal(unname(pred$errind),
                 c(0.8014108, 0.7691570, 0.8001212, 0.8150384,
                   0.8631957, 0.8150112, 0.7811111, 0.5714286),
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errind_deviance", {
    expect_equal(as.numeric(pred2$errind),
                 c(1.169316, 1.229123, 1.169316, 1.162724,
                   0.998955, 1.169598, 1.332257, 1.482081),
                 tolerance = test.tol)
})


test_that("input_groups_binomial_errall", {
    expect_equal(as.numeric(pred$errall),
                 c(0.6852769, 0.6801096, 0.6916161, 0.7209207,
                   0.6976235, 0.6707589, 0.5311111, 0.7801339),
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errall_deviance", {
    expect_equal(as.numeric(pred2$errall),
                 c(1.326829, 1.330567, 1.326829, 1.295421,
                   1.362080, 1.322675, 1.457893, 1.214764),
                 tolerance = test.tol)
})


test_that("input_groups_binomial_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.8115244, 0.7890837, 0.8138182, 0.8268031,
                   0.8716385, 0.8108259, 0.8022222, 0.6339286),
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errpre_deviance", {
    expect_equal(unname(pred2$errpre),
                 c(1.1619427, 1.2193784, 1.1619427, 1.1523452,
                   0.9970854, 1.1719199, 1.3465990, 1.4289426),
                 tolerance = test.tol)
})

###########################################################################
# Input groups, multinomial response
###########################################################################

set.seed(1234)
n=900
k=2  # of classes
p=500

scommon=10 # # of common important  features
sindiv=c(50,40,20,10,10)  #of individual important features
class.sizes=c(450, 450)
del=rep(2.5,k)
del2=rep(5, k)
means = rep(0, k)
sigma=20

out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma, outcome="multinomial", mult.classes=3)

x=out$x
y=out$y
groups=out$groups

out2=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma, outcome="multinomial", mult.classes=3)
xtest=out2$x
groupstest=out2$groups
ytest=out2$y

fit=ptLasso(x,y,groups=groups,alpha=0.9,family="multinomial",type.measure="class",foldid=NULL, nfolds=4, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest, type="class")
cvfit=cv.ptLasso(x,y,groups=groups,family="multinomial",type.measure="class",foldid=NULL, nfolds=4, overall.lambda="lambda.min")

test_that("input_groups_multinomial_errind", {
    expect_equal(unname(pred$errind),
                 c(0.3877778, 0.3877778, 0.3877778, 0.4066667, 0.3688889),
                 tolerance = test.tol)
})

test_that("input_groups_multinomial_errall_classes", {
    expect_equal(unname(pred$errall),
                 c(0.5177778, 0.5177778, 0.5177778, 0.5466667, 0.4888889),
                 tolerance = test.tol)
})

test_that("input_groups_multinomial_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.4222222, 0.4222222, 0.4222222, 0.4066667, 0.4377778),
                 tolerance = test.tol)
})

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
del=rep(2.5,k)
del2=rep(5, k)
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
                 0.2,
                 tolerance = test.tol)
})

test_that("input_groups_cox_errind", {
    expect_equal(unname(pred$errind),
                 c(0.5726963, 0.5656087, 0.5781750, 0.6404417, 0.5208463, 0.5912654, 0.5443548, 0.5311355),
                 tolerance = test.tol)
})


test_that("input_groups_cox_errall_classes", {
    expect_equal(unname(pred$errall),
                 c(0.5032823, 0.4824847, 0.4957748, 0.5314010, 0.4237710, 0.5991041, 0.3709677, 0.4871795),
                 tolerance = test.tol)
})

test_that("input_groups_cox_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.5421023, 0.5165051, 0.5414635, 0.5921325, 0.5202240, 0.5834267, 0.4032258, 0.4835165),
                 tolerance = test.tol)
})

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


fit=ptLasso(x,y,groups=groups,alpha=0.222,family="multinomial",use.case="targetGroups", type.measure="class",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)

fit2=ptLasso(x,y,groups=groups,alpha=0.222,family="multinomial",use.case="targetGroups", type.measure="deviance",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred2=predict.ptLasso(fit2,xtest,groupstest=groupstest, ytest=ytest)

cvfit = cv.ptLasso(x,y,groups=groups,family="multinomial",type.measure="class", use.case="targetGroups", foldid=NULL, nfolds=3, overall.lambda="lambda.min")
cvfit2 = cv.ptLasso(x,y,groups=groups,family="multinomial",type.measure="deviance", use.case="targetGroups", foldid=NULL, nfolds=3, overall.lambda="lambda.min")

test_that("target_groups_misclassification_alphahat", {
    expect_equal(cvfit$alphahat,
                 .6,
                 tolerance = test.tol)
})

test_that("target_groups_deviance_alphahat", {
    expect_equal(cvfit2$alphahat,
                 0.8,
                 tolerance = test.tol)
})

test_that("target_groups_misclassification_errind", {
    expect_equal(unname(pred$errind),
                 c(0.03000000, 0.05555556, 0.05333333, 0.07000000, 0.04333333),
                 tolerance = test.tol)
})

test_that("target_groups_deviance_errind", {
    expect_equal(unname(pred2$errind),
                 c(0.2287016, 0.2513482, 0.2554789, 0.2735893, 0.2249764),
                 tolerance = test.tol)
})


test_that("target_groups_misclassification_errpre", {
    expect_equal(as.numeric(pred$errpre),
                 c(0.05666667, 0.05111111, 0.05333333, 0.04666667, 0.05333333), 
                 tolerance = test.tol)
})

test_that("target_groups_deviance_errpre", {
    expect_equal(as.numeric(pred2$errpre),
                 c(0.2632167, 0.2732831, 0.2520971, 0.2998221, 0.2679302), 
                 tolerance = test.tol)
})

test_that("target_groups_misclassification_errall", {
    expect_equal(as.numeric(pred$errall),
                 c(0.04,   NA,   NA,   NA,   NA), 
                 tolerance = test.tol)
})


test_that("target_groups_deviance_errall", {
    expect_equal(as.numeric(pred2$errall),
                 c(0.2526715,   NA,   NA,   NA,   NA), 
                 tolerance = test.tol)
})


###########################################################################
# Example test 
###########################################################################
test_that("multiplication works", {
  expect_equal(2 * 2, 4)
})
