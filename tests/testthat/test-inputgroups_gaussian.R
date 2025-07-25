# To run this:
#require(testthat)
#test_file("~/Dropbox/pretrain/erin/workspace/ptLasso/test-ptLasso.R")

library(ptLasso)

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

test_that("plot.ptLasso runs without error", {
  expect_silent(plot(fit))
})

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


