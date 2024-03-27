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

pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)
pred.1se=predict(fit,xtest,groupstest=groupstest, ytest=ytest, s="lambda.1se")

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

test_that("input_groups_gaussian_suppre_common", {
    expect_equal(pred$suppre.common,
                 c( 2,   7,   9,  16,  18,  22,  27,  30,  34,  39,  44,  45,  46,  47,  49,  50,  55,  57,  60,
                   62,  63,  67,  80,  85,  86,  87,  92,  99, 100, 104, 106, 117, 118, 127, 129, 135, 137, 138, 139),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_suppre_common_1se", {
    expect_equal(pred.1se$suppre.common,
                 c( 2,   7,   9,  16,  18,  22,  27,  30,  34,  39,  44,  45,  46,  47,  49,  50,  55,  57,  60,
                   62,  63,  67,  80,  85,  86,  87,  92,  99, 100, 104, 106, 117, 118, 127, 129, 135, 137, 138, 139),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_suppre_individual", {
    expect_equal(pred$suppre.individual,
                 c( 4,   5,  11,  13,  14,  15,  17,  20,  21,  23,  24,  25,  32,  38,  40,  41,  43,  48,  51,
                   52,  53,  54,  59,  61,  64,  65,  66,  70,  71,  72,  73,  74,  77,  79,  81,  82,  83,  84,
                   88,  89,  90,  91,  95,  97,  98, 101, 105, 109, 110, 111, 114, 115, 119, 121, 122, 124, 125,
                   126, 133, 136),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_suppre_individual_1se", {
    expect_equal(pred.1se$suppre.individual,
                 c( 11,  13,  24,  32,  38,  43,  53,  66,  72,  73,  81,  84, 105, 115),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_varying_alpha", {
    expect_equal(cvfit$varying.alphahat,
                 c(0.7, 1.0, 0.8, 0.9, 0.9),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_varying_alpha_group2", {
    expect_equal(pred.cv.gp2$yhatpre,
                 pred.cv$yhatpre[groupstest == 2],
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_default_typemeasure", {
    expect_equal(pred3$erroverall,
                 pred$erroverall,
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errind", {
    expect_equal(unname(pred$errind),
                 c(1244.7535, 1112.2134, 1474.2703, 1375.4495, 1154.0878, 684.8444, 872.4152, 0.1399469),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errind_mae", {
    expect_equal(unname(pred2$errind),
                 c(26.81726, 25.54627, 28.35398, 29.66716, 24.83674, 20.92532, 23.94817, 0.1307893),
                 tolerance = test.tol)
})


test_that("input_groups_gaussian_erroverall_classes", {
    expect_equal(unname(pred$erroverall),
                 c(1557.3850, 1385.5203, 1808.8799, 1821.3583, 1389.9090,  780.7341, 1126.7373, -0.07606314),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_erroverall_classes_mae", {
    expect_equal(unname(pred2$erroverall),
                 c(30.07361, 28.13251, 33.17894, 33.70687, 26.47559, 21.75913, 25.54100, -0.04667227),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errpre", {
    expect_equal(unname(pred$errpre),
                 c(1266.0932, 1127.9857, 1517.4517, 1391.3365, 1161.3888,  683.1710, 886.5804, 0.1252024),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errpre_mae", {
    expect_equal(unname(pred2$errpre),
                 c(26.91595, 25.52184, 29.18884, 29.32310, 24.57115, 20.49902, 24.02710, 0.1406901),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_cvfit", {
    expect_equal(unname(cvfit$errpre[, "mean"]),
                 c(1220.9146, 1188.0768, 1095.7855, 1154.4937, 1063.4857, 1077.6350, 1028.0313, 1003.9437, 1052.5944, 990.6146, 948.4639),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_cvfit_mae", {
    expect_equal(unname(cvfit2$errpre[, "mean"]),
                 c(27.16340, 26.95087, 26.12019, 26.37963, 25.15874, 26.13136, 25.23187, 24.98620, 25.28564, 24.60671, 24.00805),
                 tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="gaussian",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
test_that("input_groups_gaussian_type_measure", {
    expect_equal(check.type.default$type.measure,
                 "mse")
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="gaussian",foldid=NULL, nfolds=5,type.measure="C", overall.lambda = "lambda.min")
test_that("input_groups_gaussian_wrong_type_measure", {
    expect_equal(check.type.default$type.measure,
                 "mse")
})


spl = sample(1:nrow(x), 100)
wrong.fit = cv.glmnet(x[spl, ], y[spl])
err=tryCatch( ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",type.measure="mse",foldid=foldid,
                      overall.lambda = "lambda.min", fitoverall = wrong.fit , verbose=TRUE),
             error = function(x) "error")
test_that("wrong_training_data_overall", {
    expect_equal(err, "error")
})

err=tryCatch( ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",type.measure="mse",foldid=foldid,
                      overall.lambda = "lambda.min", fitoverall = wrong.fit , verbose=TRUE),
             error = function(x) "error")
test_that("wrong_model_type_overall", {
    expect_equal(err, "error")
})

err=tryCatch( ptLasso(x,y,groups=groups,alpha=0.9,family="gaussian",type.measure="mse",foldid=foldid,
                      overall.lambda = "lambda.min", fitoverall = fit3$fitoverall, fitind = fit3$fitind[1:2], verbose=TRUE),
             error = function(x) "error")
test_that("missing_individual_models_type_overall", {
    expect_equal(err, "error")
})


bad.fitind=lapply(1:k, function(kk) cv.glmnet(x[which(groups == kk)[1:20], ], y[which(groups == kk)[1:20]], keep=TRUE))
err=tryCatch(cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=foldid,
                      overall.lambda = "lambda.min", fitind=bad.fitind),
             error = function(x) "error")
test_that("wrong_training_data_individual", {
    expect_equal(err, "error")
})


cvfit1=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=NULL, nfolds=5,
                 overall.lambda = "lambda.min", overall.gamma = "gamma.min",
                 relax=TRUE)

cvfit2=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=NULL, nfolds=5,
                 overall.lambda = "lambda.min", overall.gamma = "gamma.1se",
                 relax=TRUE)

test_that("relax_gamma_min", {
    expect_equal(unname(cvfit1$errpre[1, ]),
                 c(0.0000, 1304, 1194, 1293, 1637, 1334, 919, 814, 1265),
                 tolerance = test.tol)
})

test_that("relax_gamma_1se", {
    expect_equal(unname(cvfit2$errpre[1, ]),
                 c(0.0000, 1318, 1152, 1285, 1694, 1299, 971, 813, 982),
                 tolerance = test.tol)
})

pred.cv1=predict(cvfit1,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
pred.cv2=predict(cvfit2,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")

test_that("pred_gamma_min", {
    expect_equal(unname(pred.cv1$errpre),
                 c(1346, 1189, 1346, 1517, 1686, 1166, 688, 888),
                 tolerance = test.tol)
})

test_that("pred_gamma_1se", {
    expect_equal(unname(pred.cv2$errpre),
                 c(1362, 1376, 1362, 1249, 1603, 1157, 833, 2037),
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

fit=ptLasso(x,y,groups=groups,alpha=0.9,family="binomial",type.measure="auc",foldid=NULL, nfolds=3, overall.lambda="lambda.min")
pred=predict(fit,xtest,groupstest=groupstest, ytest=ytest)

cvfit=cv.ptLasso(x,y,groups=groups,family="binomial",type.measure="auc",foldid=NULL, nfolds=5, overall.lambda="lambda.min")
pred.cv=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
pred.cv.fixed=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest, alphatype="fixed")

pred.test=predict(cvfit,xtest,groupstest=groupstest, ytest=ytest,alpha=.6)

method.0=predict(cvfit$fit[[4]], xtest[groupstest == 2, ], groupstest=groupstest[groupstest == 2])$yhatpre
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
                 c(0.8120918, 0.7965475, 0.8208155, 0.8262916, 0.8827392, 0.8261719, 0.7522222, 0.6953125),
                 tolerance = test.tol
                 )
})

test_that("input_groups_binomial_cvfit_varying_results", {
    expect_equal(unname(pred.cv$errpre),
                 c(0.8017334, 0.7746317, 0.8026978, 0.8080818, 0.8669481, 0.8233817, 0.8066667, 0.5680804),
                 tolerance = test.tol
                 )
})

test_that("input_groups_binomial_cvfit_varying_results", {
    expect_equal(unname(pred.cv.fixed$errpre),
                 c(0.7971495, 0.7627188, 0.7942565, 0.8140153, 0.8638211, 0.7898996, 0.7777778, 0.5680804),
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
                 0.6,
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errind", {
    expect_equal(unname(pred$errind),
                 c(0.7946016, 0.7644985, 0.7946722, 0.8066496, 0.8647592, 0.8007812, 0.7822222, 0.5680804),
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errind_deviance", {
    expect_equal(as.numeric(pred2$errind),
                 c(1.183702, 1.238977, 1.183702, 1.168604, 1.036093, 1.188568, 1.341514, 1.460107),
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
                 c(1.326829, 1.330567, 1.326829, 1.295421,
                   1.362080, 1.322675, 1.457893, 1.214764),
                 tolerance = test.tol)
})


test_that("input_groups_binomial_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.8083980, 0.7777321, 0.8084310, 0.8298721, 0.8747655, 0.8013393, 0.7788889, 0.6037946),
                 tolerance = test.tol)
})

test_that("input_groups_binomial_errpre_deviance", {
    expect_equal(unname(pred2$errpre),
                 c(1.187135, 1.256267, 1.187135, 1.160557, 1.023357, 1.176455, 1.398885, 1.522081),
                 tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="binomial",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")
test_that("input_groups_binomial_type_measure", {
    expect_equal(check.type.default$type.measure,
                 "deviance")
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
                 c(0.4055556, 0.4055556, 0.4055556, 0.3866667, 0.4244444),
                 tolerance = test.tol)
})

test_that("input_groups_multinomial_erroverall_classes", {
    expect_equal(unname(pred$erroverall),
                 c(0.5177778, 0.5177778, 0.5177778, 0.5466667, 0.4888889),
                 tolerance = test.tol)
})

test_that("input_groups_multinomial_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.3900000, 0.3900000, 0.3900000, 0.3955556, 0.3844444),
                 tolerance = test.tol)
})

check.type.default=cv.ptLasso(x,y,groups=groups,family="multinomial",foldid=NULL, nfolds=5, overall.lambda = "lambda.min")

test_that("input_groups_multinomial_type_measure", {
    expect_equal(check.type.default$type.measure,
                 "deviance")
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
                 0.7,
                 tolerance = test.tol)
})

test_that("input_groups_cox_errind", {
    expect_equal(unname(pred$errind),
                 c(0.5581741, 0.5685027, 0.5742311, 0.6417018, 0.5108565, 0.5510662, 0.6388889, 0.5000000),
                 tolerance = test.tol)
})


test_that("input_groups_cox_erroverall_classes", {
    expect_equal(unname(pred$erroverall),
                 c(0.5432290, 0.5234780, 0.5332552, 0.5428973, 0.4825090, 0.6442200, 0.4087302, 0.5390335),
                 tolerance = test.tol)
})

test_that("input_groups_cox_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.5420089, 0.5415226, 0.5582235, 0.5798172, 0.5211098, 0.6531987, 0.4404762, 0.5130112),
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

#fitoverall = cv.glmnet(x, y, family = "cox", keep = TRUE, type.measure="C")
#fit=ptLasso(x[groups %in% (1:2), ],y[groups %in% (1:2), ],groups=groups[groups %in% (1:2)],alpha=0.1,family="cox",type.measure="C",foldid=NULL, nfolds=5, overall.lambda="lambda.min", fitoverall=fitoverall, group.intercepts=FALSE)
#pred=predict(fit,xtest[groupstest ==1 ,],groupstest=groupstest[groupstest == 1], ytest=ytest[groupstest==1,])

#cvfit = cv.ptLasso(x,y,groups=groups,family="cox",type.measure="C",foldid=NULL, nfolds=5, overall.lambda="lambda.min", fitoverall=fitoverall, group.intercepts=FALSE)

#y = Surv(runif(nrow(x), 0, .1), y[, 1], y[, 2])
#fit=ptLasso(x, y, groups=groups, alpha=0.1,family="cox",type.measure="C",foldid=NULL, nfolds=5, overall.lambda="lambda.min")

#ytest = Surv(runif(nrow(xtest), 0, .1), ytest[, 1], ytest[, 2])
#pred = predict(fit, xtest, groupstest, ytest)
#test_that("three_arg_Surv", {
#    expect_equal(unname(pred$errpre),
#                 c(0.5468076, 0.5308165, 0.5540374, 0.5840366, 0.5174910, 0.6610550, 0.4007937, 0.4907063))
#})

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
pred2=predict(cvfit2,xtest,groupstest=groupstest, ytest=ytest)
pred3=predict(cvfit2,xtest,groupstest=groupstest, ytest=ytest, alphatype = "varying")


test_that("target_groups_misclassification_alphahat", {
    expect_equal(cvfit$alphahat,
                 0.4,
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
                 c(0.03333333, 0.05000000, 0.05000000, 0.05000000, 0.05000000), 
                 tolerance = test.tol)
})

test_that("target_groups_deviance_errpre", {
    expect_equal(as.numeric(pred2$errpre),
                 c(0.2459651, 0.2565011, 0.2619418, 0.2756477, 0.2319137), 
                 tolerance = test.tol)
})

test_that("target_groups_misclassification_erroverall", {
    expect_equal(as.numeric(pred$erroverall),
                 c(0.04,   NA,   NA,   NA,   NA), 
                 tolerance = test.tol)
})


test_that("target_groups_deviance_erroverall", {
    expect_equal(as.numeric(pred2$erroverall),
                 c(0.2526715,   NA,   NA,   NA,   NA), 
                 tolerance = test.tol)
})


test_that("target_groups_misclassification_varying_alpha", {
    expect_equal(unname(pred3$errpre),
                 c(0.2380190, 0.2523420, 0.2579512, 0.2734330, 0.2256416), 
                 tolerance = test.tol)
})


test_that("target_groups_misclassification_varying_alpha_errind_same", {
    expect_equal(unname(pred2$errind),
                 unname(pred3$errind), 
                 tolerance = test.tol)
})

###########################################################################
# Example test 
###########################################################################
test_that("multiplication works", {
  expect_equal(2 * 2, 4)
})

