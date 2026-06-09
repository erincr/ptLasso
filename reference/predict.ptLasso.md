# Predict using a ptLasso object.

Return predictions and performance measures for a test set.

## Usage

``` r
# S3 method for class 'ptLasso'
predict(
  object,
  xtest,
  groupstest = NULL,
  ytest = NULL,
  type = c("link", "response", "class"),
  s = "lambda.min",
  gamma = "gamma.min",
  return.link = FALSE,
  ...
)
```

## Arguments

- object:

  Fitted `"ptLasso"` object.

- xtest:

  Input matrix, matching the form used by `"ptLasso"` for model
  training.

- groupstest:

  A vector indicating to which group each observation belongs. Coding
  should match that used for model training. Will be NULL for target
  grouped data.

- ytest:

  Response variable. Optional. If included, `"predict"` will compute
  performance measures for xtest using `"type.measure"` from the cvfit
  object.

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

  The value(s) of alpha used to generate predictions. Will be the same
  alpha used to in model training.

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

  If `ytest` is supplied, the string name of the computed performance
  measure.

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

  If`return.link` is TRUE, return the linear link from the overall
  model.

- linkind:

  If`return.link` is TRUE, return the linear link from the individual
  models.

- linkpre:

  If`return.link` is TRUE, return the linear link from the pretrained
  models.

## Examples

``` r
# Gaussian example
set.seed(1234)
out = gaussian.example.data()
x = out$x; y=out$y; groups = out$group

outtest = gaussian.example.data()
xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups

fit = ptLasso(x, y, groups = groups, alpha = 0.5, family = "gaussian", type.measure = "mse")
pred = predict(fit, xtest, groupstest, ytest=ytest)
pred
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest) 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2 group_3 group_4 group_5    r^2
#> Overall        755.7 755.7   836.0   554.9   565.4   777.9  1044.0 0.5371
#> Pretrain       503.2 503.2   550.6   443.3   553.5   505.6   462.9 0.6918
#> Individual     532.8 532.8   584.1   443.2   567.2   550.5   518.9 0.6736
#> 
#> Support size:
#>                                          
#> Overall    64                            
#> Pretrain   94 (21 common + 73 individual)
#> Individual 109                           
```
