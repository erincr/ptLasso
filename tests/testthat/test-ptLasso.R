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
means=sample(1:k)
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
                   88,  89,  90,  91,  97,  98, 101, 105, 109, 110, 111, 112, 114, 115, 119, 121, 122, 124, 125,
                   126, 133, 136),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_suppre_individual_1se", {
    expect_equal(pred.1se$suppre.individual,
                 c( 11,  13,  24,  32,  43,  53,  66,  72,  73,  81,  84, 90, 97, 105, 115, 121),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_varying_alpha", {
    expect_equal(cvfit$varying.alphahat,
                 c(0.5, 0.7, 1.0, 0.8, 0.8),
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
                 c(27.564, 25.887, 31.110, 29.397, 24.616, 20.364, 23.948, 0.108),
                 tolerance = test.tol)
})


test_that("input_groups_gaussian_erroverall_classes", {
    expect_equal(unname(pred$erroverall),
                 c(1557.3850, 1385.5203, 1808.8799, 1821.3583, 1389.9090,  780.7341, 1126.7373, -0.07606314),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_erroverall_classes_mae", {
    expect_equal(unname(pred2$erroverall),
                 c(29.544, 27.364, 33.133, 32.903, 26.473, 20.364, 23.948, -0.012),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errpre", {
    expect_equal(unname(pred$errpre),
                 c(1254.6427787, 1125.4215336, 1528.4164546, 1319.1309333, 1154.4635033,
                   738.5164069,  886.5803697,    0.1331139),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_errpre_mae", {
    expect_equal(unname(pred2$errpre),
                 c(27.553, 25.883, 31.052, 29.394, 24.667, 20.359, 23.942,  0.107),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_cvfit", {
    expect_equal(unname(cvfit$errpre[, "mean"]),
                 c(1144.6457, 1093.0088, 1071.6828, 1082.9672, 1009.2513,  984.4556,  995.1516,
                   955.1581,  936.4629,  973.1961, 1040.0583),
                 tolerance = test.tol)
})

test_that("input_groups_gaussian_cvfit_mae", {
    expect_equal(unname(cvfit2$errpre[, "mean"]),
                 c(25.579, 25.147,25.072, 24.448, 24.617, 24.657, 25.033, 24.619, 24.344, 24.934, 25.091),
                 tolerance = test.tol)
})

#################################
# Check errors
#################################
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


#################################
# Relax = true
#################################
cvfit1=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=NULL, nfolds=5,
                 overall.lambda = "lambda.min", overall.gamma = "gamma.min",
                 relax=TRUE)

cvfit2=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=NULL, nfolds=5,
                 overall.lambda = "lambda.min", overall.gamma = "gamma.1se",
                 relax=TRUE)
 

test_that("relax_gamma_min", {
    expect_equal(unname(cvfit1$errpre[1, ]),
                 c(0.000, 1175, 1137, 1175, 1238, 1291, 1028, 697, 1431),
                 tolerance = test.tol)
})

test_that("relax_gamma_1se", {
    expect_equal(unname(cvfit2$errpre[1, ]),
                 c(0.000, 1249, 1195, 1249, 1443, 1272, 1030, 780, 1449),
                 tolerance = test.tol)
})

pred.cv1=predict(cvfit1,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")
pred.cv2=predict(cvfit2,xtest,groupstest=groupstest, ytest=ytest, alphatype="varying")

test_that("pred_gamma_min", {
    expect_equal(unname(pred.cv1$errpre),
                 c(1525, 1321, 1525, 2164, 1384, 1283, 898, 878),
                 tolerance = test.tol)
})

test_that("pred_gamma_1se", {
    expect_equal(unname(pred.cv2$errpre),
                 c(1443, 1233, 1443, 2071, 1366, 1155,  688,  885),
                 tolerance = test.tol)
})

# Not sure what to test here:
groups[groups == 2] = 10
cvfit = cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse", overall.lambda = "lambda.min")
groups[groups == 10] = 2
test_that("non-contiguous_groups", {
    expect_equal(colnames(cvfit$errpre),
                 c("alpha", "overall", "mean", "wtdMean", "group_1", "group_3", "group_4", "group_5", "group_10")
                 )
})

#################################
# String groups
#################################
groups.new = rep(NA, length(groups))
groupstest.new = rep(NA, length(groupstest))
groups.new[groups == 1] = "a"
groups.new[groups == 2] = "b"
groups.new[groups == 3] = "c"
groups.new[groups == 4] = "d"
groups.new[groups == 5] = "e"

groupstest.new[groupstest == 1] = "a"
groupstest.new[groupstest == 2] = "b"
groupstest.new[groupstest == 3] = "c"
groupstest.new[groupstest == 4] = "d"
groupstest.new[groupstest == 5] = "e"
set.seed(1234)
cvfit.string=cv.ptLasso(x,y,groups=groups.new,family="gaussian",type.measure="mse", foldid=foldid, nfolds=5,
                        overall.lambda = "lambda.min")
set.seed(1234)
cvfit=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse", foldid=foldid, nfolds=5,
                        overall.lambda = "lambda.min")

test_that("string_groups", {
    expect_equal(unname(cvfit.string$errpre),
                 unname(cvfit$errpre),
                 tolerance = test.tol)
})



#################################
# Non-contiguous groups
#################################
set.seed(1234)
n=500; k=5; p=140

scommon=10; sindiv=rep(10, 5)  #of individual important features
class.sizes=rep(100, 5)
del=rep(2.5,k); del2=rep(5, k)
means = sample(1:k); sigma=20

out=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma)

x=out$x; y=out$y; groups=out$groups

nfolds=5
foldid = rep(1, nrow(x))  
for(kk in 1:k) foldid[groups == kk] = sample(1:nfolds, class.sizes[kk], replace=TRUE)

out2=makedata(n=n, p=p, k=k, scommon=scommon, sindiv=sindiv,
             class.sizes=class.sizes, beta.common=del, beta.indiv=del2,
             intercepts=means, sigma=sigma)
xtest=out2$x; groupstest=out2$groups; ytest=out2$y

groups.new = groups
groupstest.new = groupstest
groups.new[groups == 2] = 3
groups.new[groups == 3] = 4
groups.new[groups == 4] = 5
groups.new[groups == 5] = 6

set.seed(1234)
cvfit.jumbled=cv.ptLasso(x,y,groups=groups.new,family="gaussian",type.measure="mse",foldid=foldid, nfolds=5,
                        overall.lambda = "lambda.min")
set.seed(1234)
cvfit=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=foldid, nfolds=5,
                        overall.lambda = "lambda.min")

test_that("non_contiguous_groups", {
    expect_equal(unname(cvfit.jumbled$errpre),
                 unname(cvfit$errpre),
                 tolerance = test.tol)
})


groups.new = groups - 2
groupstest.new = groupstest - 2

set.seed(1234)
cvfit.jumbled=cv.ptLasso(x,y,groups=groups.new,family="gaussian",type.measure="mse",foldid=foldid, nfolds=5,
                         overall.lambda = "lambda.min")
preds.jumbled = predict(cvfit.jumbled, xtest, groupstest=groupstest.new, ytest=ytest)
preds.jumbled.subset = predict(cvfit.jumbled, xtest[groupstest %in% c(1, 3), ],
                               groupstest=groupstest.new[groupstest %in% c(1, 3)], ytest=ytest[groupstest %in% c(1, 3)])

set.seed(1234)
cvfit=cv.ptLasso(x,y,groups=groups,family="gaussian",type.measure="mse",foldid=foldid, nfolds=5,
                        overall.lambda = "lambda.min")
preds = predict(cvfit, xtest, groupstest, ytest)

test_that("negative_groups", {
    expect_equal(unname(cvfit.jumbled$errpre),
                 unname(cvfit$errpre),
                 tolerance = test.tol)
})

test_that("negative_groups_prednames", {
    expect_equal(names(preds.jumbled$errpre),
                 c("allGroups", "mean", "group_-1",  "group_0", "group_1", "group_2", "group_3", "r^2")
                 )
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
                 c(0.453, 0.453, 0.453, 0.404, 0.502),
                 tolerance = test.tol)
})

test_that("input_groups_multinomial_erroverall_classes", {
    expect_equal(unname(pred$erroverall),
                 c(0.523, 0.523, 0.523, 0.547, 0.500),
                 tolerance = test.tol)
})

test_that("input_groups_multinomial_errpre", {
    expect_equal(unname(pred$errpre),
                 c(0.382, 0.382, 0.382, 0.391, 0.373),
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


###########################################################################
# Multiresponse
###########################################################################

set.seed(1234)
n = 1000; ntrain = 500;
p = 500
sigma = 2
          
x = matrix(rnorm(n*p), n, p)
beta1 = c(rep(1, 5), rep(0.5, 5), rep(0, p - 10))
beta2 = c(rep(1, 5), rep(0, 5), rep(0.5, 5), rep(0, p - 15))
     
mu = cbind(x %*% beta1, x %*% beta2)
y  = cbind(mu[, 1] + sigma * rnorm(n), 
                mu[, 2] + sigma * rnorm(n))
     
xtest = x[-(1:ntrain), ]
ytest = y[-(1:ntrain), ]

x = x[1:ntrain, ]
y = y[1:ntrain, ]
     
# Now, we can fit a ptLasso multiresponse model:
#fit = ptLassoMult(x, y, alpha = 0.5, type.measure = "mse")
fit = ptLasso(x, y, alpha = 0.5, type.measure = "mse", use.case = "multiresponse")
# plot(fit) # to see all of the cv.glmnet models trained
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_suppre.common", {
    expect_equal(unname(preds$suppre.common),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 140, 257, 296, 346), 
                 tolerance = test.tol)
})

test_that("multresponse_suppre.individual", {
    expect_equal(unname(preds$suppre.individual),
                 c(52, 60, 120, 153, 322, 358, 365, 400, 404, 498), 
                 tolerance = test.tol)
})

test_that("multresponse_supoverall", {
    expect_equal(unname(preds$supoverall),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   52, 60, 64, 69, 99, 112, 119, 120, 125, 139, 140,
                   143, 153, 172, 179, 193, 199, 206, 223, 225, 233,
                   257, 296, 306, 313, 315, 322, 346, 357, 358, 365,
                   400, 404, 417, 418, 419, 440, 446, 457, 464, 480, 498), 
                 tolerance = test.tol)
})


test_that("multresponse_supind", {
    expect_equal(unname(preds$supind),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   21, 52, 60, 64, 69, 84, 89, 92, 99, 119, 120, 125,
                   130, 135, 140, 153, 171, 172, 179, 193, 199, 206,
                   215, 222, 223, 225, 233, 236, 239, 250, 252, 257,
                   265, 282, 294, 296, 301, 306, 311, 313, 315, 322,
                   342, 346, 357, 358, 365, 372, 378, 400, 402, 404,
                   412, 417, 419, 428, 440, 446, 448, 449, 455, 456,
                   464, 480, 498), 
                 tolerance = test.tol)
})

test_that("multresponse_errpre", {
    expect_equal(unname(preds$errpre),
                 c(9.022075, 4.511038, 4.144121, 4.877954), 
                 tolerance = test.tol)
})

test_that("multresponse_errind", {
    expect_equal(unname(preds$errind),
                 c(9.465296, 4.732648, 4.243210, 5.222087), 
                 tolerance = test.tol)
})

test_that("multresponse_errall", {
    expect_equal(unname(preds$erroverall),
                 c(9.394191, 4.697096, 4.226602, 5.167589), 
                 tolerance = test.tol)
})


set.seed(1234)
fit = ptLasso(x, y, alpha = 0.5, type.measure = "mae", use.case = "multiresponse")
# plot(fit) # to see all of the cv.glmnet models trained
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_mae_suppre.common", {
    expect_equal(unname(preds$suppre.common),
                 c( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                   14, 15, 120, 140, 172, 257, 296, 346, 365, 400, 464), 
                 tolerance = test.tol)
})

test_that("multresponse_mae_suppre.individual", {
    expect_equal(unname(preds$suppre.individual),
                 c(52, 60, 153, 322, 358, 404, 498), 
                 tolerance = test.tol)
})

test_that("multresponse_mae_supoverall", {
    expect_equal(unname(preds$supoverall),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                   14, 15, 52, 60, 64, 69, 99, 112, 119, 120,
                   125, 139, 140, 143, 153, 172, 179, 193, 199,
                   206, 223, 225, 233, 257, 296, 306, 313, 315,
                   322, 346, 357, 358, 365, 400, 404, 417, 418,
                   419, 440, 446, 457, 464, 480, 498), 
                 tolerance = test.tol)
})


test_that("multresponse_mae_supind", {
    expect_equal(unname(preds$supind),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                   15, 21, 52, 60, 64, 69, 84, 89, 92, 99, 119,
                   120, 125, 130, 135, 140, 153, 171, 172, 179,
                   193, 199, 206, 215, 222, 223, 225, 233, 236,
                   239, 250, 252, 257, 265, 282, 294, 296, 301,
                   306, 311, 313, 315, 322, 342, 346, 357, 358,
                   365, 372, 378, 400, 402, 404, 412, 417, 419,
                   428, 440, 446, 448, 449, 455, 456, 464, 480, 498), 
                 tolerance = test.tol)
})

test_that("multresponse_mae_errpre", {
    expect_equal(unname(preds$errpre),
                 c(3.425577, 1.712789, 1.612142, 1.813435), 
                 tolerance = test.tol)
})

test_that("multresponse_mae_errind", {
    expect_equal(unname(preds$errind),
                 c(3.474751, 1.737375, 1.622945, 1.851805), 
                 tolerance = test.tol)
})

test_that("multresponse_mae_errall", {
    expect_equal(unname(preds$erroverall),
                 c(3.463764, 1.731882, 1.622082, 1.841682), 
                 tolerance = test.tol)
})


set.seed(1234)
fit = cv.ptLasso(x, y, type.measure = "mae", use.case = "multiresponse")
preds = predict(fit, xtest, ytest=ytest) # to predict on new data

test_that("multresponse_cv_suppre.common", {
    expect_equal(unname(preds$suppre.common),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                   13, 14, 15, 120, 140, 172, 257, 296, 346,
                   365, 400, 464), 
                 tolerance = test.tol)
})

test_that("multresponse_cv_suppre.individual", {
    expect_equal(unname(preds$suppre.individual),
                 c(60, 153, 322), 
                 tolerance = test.tol)
})

test_that("multresponse_cv_supoverall", {
    expect_equal(unname(preds$supoverall),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                   15, 52, 60, 64, 69, 99, 112, 119, 120, 125, 139,
                   140, 143, 153, 172, 179, 193, 199, 206, 223, 225,
                   233, 257, 296, 306, 313, 315, 322, 346, 357, 358,
                   365, 400, 404, 417, 418, 419, 440, 446, 457, 464, 480, 498), 
                 tolerance = test.tol)
})


test_that("multresponse_cv_supind", {
    expect_equal(unname(preds$supind),
                 c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                   15, 21, 52, 60, 64, 69, 84, 89, 92, 99, 119,
                   120, 125, 130, 135, 140, 153, 171, 172, 179,
                   193, 199, 206, 215, 222, 223, 225, 233, 236,
                   239, 250, 252, 257, 265, 282, 294, 296, 301,
                   306, 311, 313, 315, 322, 342, 346, 357, 358,
                   365, 372, 378, 400, 402, 404, 412, 417, 419,
                   428, 440, 446, 448, 449, 455, 456, 464, 480, 498), 
                 tolerance = test.tol)
})

test_that("multresponse_cv_errpre", {
    expect_equal(unname(preds$errpre),
                 c(3.422234, 1.711117, 1.623132, 1.799102), 
                 tolerance = test.tol)
})

test_that("multresponse_cv_errind", {
    expect_equal(unname(preds$errind),
                 c(3.474751, 1.737375, 1.622945, 1.851805), 
                 tolerance = test.tol)
})

test_that("multresponse_cv_errall", {
    expect_equal(unname(preds$erroverall),
                 c(3.463764, 1.731882, 1.622082, 1.841682), 
                 tolerance = test.tol)
})


###########################################################################
# Example test 
###########################################################################
test_that("multiplication works", {
  expect_equal(2 * 2, 4)
})

