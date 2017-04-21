## My Solutions to Exercises:

### 1. What Linear Regression training algorithm can you use if you have a training set with millions of features?

> Using gradient descent for models that has more than 10,000 features. specifically, use Stochastic GD or mini-batch when training set doesn't fit into memory, batch GD can be applied when fit. Normal equation is not practical since the complexity is around O(n^3)

### 2. Suppose the features in your training set have very different scales. What algorithms might suffer from this, and how? What can you do about it?

> Gradient Descent needs feature scaling. It will take a longer time if data are not normalized. usually use Mean Normalization. while normal Equation doesn't have this problem

### 3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?

> Not at all since the logistics curve's loss function is convex. GD will finally reach the optimal solution.

### 4. Do all Gradient Descent algorithms lead to the same model provided you let them run long enough?

> No. SGD is erratic, don't know when they'll get the optimal solution. Traing SGD for a long time doesn't mean it gradually get nearer to the target. sometime bounce up &down, not truly converge, it will jump back & forth  

### 5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation
error consistently goes up, what is likely going on? How can you fix this

> overfitting the test case results in low generalization capability. cross-validation. feed more data,
Another possible case is that both validation error & training error goes up. should reduce learning rate.

### 6. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?

> nope. both SGD & mini-batch GD's performace are not stable, they fluctuate all the time. doesn't gurantee progresss  

### 7. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?

> mini-batch GD is the fastest GD, Batch GD will actually converge. learning schedule, early stopping

### 8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the
training error and the validation error. What is happening? What are three ways to solve this?

> overfitting - when the training error is also large; 1. feed more data, 2. regularize, 3. decrease degree

### 9. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly
high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α
or reduce it?

> underfitting(when training error is high): reduce hyperparameter to relive bias

### 10. Why would you want to use:
Ridge Regression instead of Linear Regression?
Lasso instead of Ridge Regression?
Elastic Net instead of Lasso?

> 
*Ridge, by default; 
*Lasso when assuming there're less important features; 
*Elastic Net: #feature > #instance or features are mutally correlated 

### 11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression
classifiers or one Softmax Regression classifier?

> 2 logistic regression classifiers since Softmax Regression focus on  1 type of data at 1 time

### 12. Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).
