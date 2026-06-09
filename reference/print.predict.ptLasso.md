# Print the predict.ptLasso object.

Print the predict.ptLasso object.

## Usage

``` r
# S3 method for class 'predict.ptLasso'
print(x, ...)
```

## Arguments

- x:

  output of predict called with a ptLasso object.

- ...:

  other arguments to pass to the print function.

## See also

`ptLasso` and `predict.ptLasso`.

## Author

Erin Craig and Rob Tibshirani  
Maintainer: Erin Craig \<erincr@stanford.edu\>

## Examples

``` r
# Train data
out = gaussian.example.data(k=2, class.sizes = c(50, 50))
x = out$x; y=out$y; groups = out$group;

# Test data
outtest = gaussian.example.data(k=2, class.sizes = c(50, 50))
xtest=outtest$x; ytest=outtest$y; groupstest=outtest$groups

fit = ptLasso(x, y, groups = groups, nfolds = 3, family = "gaussian", type.measure = "mse")
pred = predict(fit, xtest, groupstest, ytest=ytest, s="lambda.min")
print(pred)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, s = "lambda.min") 
#> 
#> 
#> alpha =  0.5 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2       r^2
#> Overall        583.1 583.1   537.9   628.2 -0.018832
#> Pretrain       573.1 573.1   554.9   591.4 -0.001435
#> Individual     589.9 589.9   548.5   631.3 -0.030719
#> 
#> Support size:
#>                                         
#> Overall    0                            
#> Pretrain   26 (0 common + 26 individual)
#> Individual 30                           

# If ytest is not supplied, just prints the pretrained predictions.
pred = predict(fit, xtest, groupstest, s="lambda.min")
print(pred)
#> 
#> Call:  
#> predict.ptLasso(object = fit, xtest = xtest, groupstest = groupstest,  
#>     s = "lambda.min") 
#> 
#> 
#> alpha =  0.5 
#> 
#> Support size:
#>                                         
#> Overall    0                            
#> Pretrain   26 (0 common + 26 individual)
#> Individual 30                           
```
