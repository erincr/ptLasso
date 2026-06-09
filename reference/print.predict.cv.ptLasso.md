# Print the predict.cv.ptLasso object.

Print the predict.cv.ptLasso object.

## Usage

``` r
# S3 method for class 'predict.cv.ptLasso'
print(x, ...)
```

## Arguments

- x:

  output of predict called with a ptLasso object.

- ...:

  other arguments to pass to the print function.

## See also

`cv.ptLasso` and `predict.cv.ptLasso`.

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

cvfit = cv.ptLasso(x, y, groups = groups, nfolds = 3, family = "gaussian", type.measure = "mse")
pred = predict(cvfit, xtest, groupstest, ytest=ytest, s="lambda.min")
print(pred)
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     ytest = ytest, s = "lambda.min") 
#> 
#> 
#> alpha =  0.7 
#> 
#> Performance (Mean squared error):
#> 
#>            allGroups  mean group_1 group_2       r^2
#> Overall        547.9 547.9   541.6   554.2  0.152982
#> Pretrain       581.5 581.5   564.3   598.8  0.100969
#> Individual     648.1 648.1   595.2   700.9 -0.001881
#> 
#> Support size:
#>                                         
#> Overall    17                           
#> Pretrain   12 (0 common + 12 individual)
#> Individual 2                            

# If ytest is not supplied, just prints the pretrained predictions.
pred = predict(cvfit, xtest, groupstest, s="lambda.min")
print(pred)
#> 
#> Call:  
#> predict.cv.ptLasso(object = cvfit, xtest = xtest, groupstest = groupstest,  
#>     s = "lambda.min") 
#> 
#> 
#> alpha =  0.7 
#> 
#> Support size:
#>                                         
#> Overall    17                           
#> Pretrain   12 (0 common + 12 individual)
#> Individual 2                            
```
