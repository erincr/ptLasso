# Predict using a cv.ptLasso object.

Return predictions and performance measures for a test set.

## Usage

``` r
# S3 method for class 'cv.ptLasso'
predict(
  object,
  xtest,
  groupstest = NULL,
  ytest = NULL,
  alpha = NULL,
  alphatype = c("fixed", "varying"),
  type = c("link", "response", "class"),
  s = "lambda.min",
  gamma = "gamma.min",
  return.link = FALSE,
  ...
)
```

## Arguments

- object:

  Fitted `"cv.ptLasso"` object.

- xtest:

  Input matrix, matching the form used by `"cv.ptLasso"` for model
  training.

- groupstest:

  A vector indicating to which group each observation belongs. Coding
  should match that used for model training. Will be NULL for target
  grouped data.

- ytest:

  Response variable. Optional. If included, `"predict"` will compute
  performance measures for xtest using `"type.measure"` from the cvfit
  object.

- alpha:

  The chosen alpha to use for prediction. May be a vector containing one
  value of alpha for each group. If NULL, this will rely on the choice
  of "alphatype".

- alphatype:

  Choice of '"fixed"' or '"varying"'. If '"fixed"', use the alpha that
  achieved best cross-validated performance. If '"varying"', each group
  uses the alpha that optimized the group-specific cross-validated
  performance.

- type:

  Type of prediction required. Type '"link"' gives the linear predictors
  for '"binomial", '"multinomial"' or '"cox"' models; for '"gaussian"'
  models it gives the fitted values. Type '"response"' gives the fitted
  probabilities for '"binomial"' or '"multinomial"', and the fitted
  relative-risk for '"cox"'; for '"gaussian"' type '"response"' is
  equivalent to type '"link"'. Note that for '"binomial"' models,
  results are returned only for the class corresponding to the second
  level of the factor response. Type '"class"' applies only to
  '"binomial"' or '"multinomial"' models, and produces the class label
  corresponding to the maximum probability.

- s:

  Value of the penalty parameter 'lambda' at which predictions are
  required. Will use the same lambda for all models; can be a numeric
  value, '"lambda.min"' or '"lambda.1se"'. Default is '"lambda.min"'.

- gamma:

  For use only when 'relax = TRUE' was specified during training. Value
  of the penalty parameter 'gamma' at which predictions are required.
  Will use the same gamma for all models; can be a numeric value,
  '"gamma.min"' or '"gamma.1se"'. Default is '"gamma.min"'.

- return.link:

  If `TRUE`, will additionally return the linear link for the overall,
  pretrained and individual models: `linkoverall`, `linkpre` and
  `linkind`.

- ...:

  other arguments to be passed to the `"predict"` function.

## Value

A list containing the requested predictions. If `ytest` is included,
will also return error measures.

- call:

  The call that produced this object.

- alpha:

  The value(s) of alpha used to generate predictions.

- yhatoverall:

  Predictions from the overall model.

- yhatind:

  Predictions from the individual models.

- yhatpre:

  Predictions from the pretrained models.

- supoverall:

  Indices of the features selected by the overall model.

- supind:

  Union of the indices of the features selected by the individual
  models.

- suppre.common:

  Features selected in the first stage of pretraining.

- suppre.individual:

  Union of the indices of the features selected by the pretrained
  models, without the features selected in the first stage.

- type.measure:

  If `ytest` is supplied, the performance measure computed.

- erroverall:

  If `ytest` is supplied, performance for the overall model. This is a
  named vector containing performance for (1) the entire dataset, (2)
  the average performance across groups, (3) the average performance
  across groups weighted by group size and (4) group-specific
  performance.

- errind:

  If `ytest` is supplied, performance for the overall model. As
  described in `erroverall`.

- errpre:

  If `ytest` is supplied, performance for the overall model. As
  described in `erroverall`.

- linkoverall:

  If `return.link` is TRUE, return the linear link from the overall
  model.

- linkind:

  If `return.link` is TRUE, return the linear link from the individual
  models.

- linkpre:

  If `return.link` is TRUE, return the linear link from the pretrained
  models.

## See also

`ptLasso`, `cv.ptLasso` and `predict.cv.ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
#### Gaussian example
set.seed(1234)
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group;
outtest = gaussian.example.data(k=2, class.sizes = c(50, 50))
xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups;

# Model fitting
# By default, use the single value of alpha that had the best CV performance on the entire dataset:
cvfit = cv.ptLasso(x, y, groups = groups, family = "gaussian", type.measure = "mse")
pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
pred
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, s = "lambda.min") 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2    r^2
#> Overall        609.0 609.0   548.3   669.7 0.2330
#> Pretrain       650.8 650.8   601.1   700.6 0.1803
#> Individual     675.2 675.2   667.6   682.8 0.1496
#> 
#> Support size:
#>                                         
#> Overall    21                           
#> Pretrain   18 (6 common + 12 individual)
#> Individual 26                           

# For each group, use the value of alpha that had the best CV performance for that group:
pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min", alphatype = "varying")
pred
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, alphatype = "varying", s = "lambda.min") 
#> 
#> 
#> 
#> alpha:
#> [1] 0.1 1.0
#> 
#> 
#> Performance (Mean squared error):
#>            overall  mean wtdMean group_1 group_2
#> Overall      609.0 609.0   609.0   548.3   669.7
#> Pretrain     629.7 629.7   629.7   576.6   682.8
#> Individual   675.2 675.2   675.2   667.6   682.8
#> 
#> 
#> Support size:
#>                                         
#> Overall    21                           
#> Pretrain   26 (6 common + 20 individual)
#> Individual 26                           

# Specify a single value of alpha and use lambda.1se.
pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.1se",
               alphatype = "varying", alpha = .3)
pred
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, alpha = 0.3, alphatype = "varying", s = "lambda.1se") 
#> 
#> 
#> 
#> alpha =  0.3 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2     r^2
#> Overall        709.1 709.1   573.8   844.5 0.10690
#> Pretrain       660.6 660.6   585.7   735.5 0.16800
#> Individual     743.8 743.8   667.6   820.0 0.06322
#> 
#> Support size:
#>                                       
#> Overall    6                          
#> Pretrain   7 (6 common + 1 individual)
#> Individual 10                         

# Specify a vector of choices for alpha: 
pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min",
               alphatype = "varying", alpha = c(.1, .5))
pred
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, alpha = c(0.1, 0.5), alphatype = "varying",  
#>     s = "lambda.min") 
#> 
#> 
#> alpha:
#> [1] 0.1 0.5
#> 
#> 
#> Performance (Mean squared error):
#>            overall  mean wtdMean group_1 group_2
#> Overall      609.0 609.0   609.0   548.3   669.7
#> Pretrain     638.6 638.6   638.6   576.6   700.6
#> Individual   675.2 675.2   675.2   667.6   682.8
#> 
#> 
#> Support size:
#>                                         
#> Overall    21                           
#> Pretrain   18 (6 common + 12 individual)
#> Individual 26                           
```
