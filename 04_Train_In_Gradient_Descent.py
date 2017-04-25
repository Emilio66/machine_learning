import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt

# generate data randomly
rnd.seed(42)
X = 2 * rnd.rand(100, 1)
y = 4 + 3 * X + rnd.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]# add b = 1 to each instance
X_new = np.array([[0], [2]])  # 2 new instances for testing
X_new_b = np.c_[np.ones((2, 1)), X_new]  
print(X_new_b)
print(X)

## Batch Gradient Descent
theta_path_bgd = []
m = float(len(X_b)) # incase the value is 0
def plot_gradient_descent(theta, eta, theta_path=None):
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
	    #print("Predict Results: ", y_predict, " Theta: ", theta)
            style = "g-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # mean square error's gradient/derivative
	#print(1/m, X_b.T, X_b.dot(theta) - y)
	#print("before: ", theta, "eta: ", eta)
        theta = theta - eta * gradients     # update parameter by gradients
	#print("after: ", theta, " gradients:", gradients)
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

rnd.seed(42)
theta = rnd.randn(2,1)  # random initialize the startpoint of gradient descent as 2x1 matrix

plt.figure(figsize=(20,8))
plt.suptitle('Linear Regression with Batch Gradient Descent', fontsize=20)
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.55)
print('Batch Gradient Descent')
plt.show()
print('Theta Path Size: ', len(theta_path_bgd))

# define a function to decrease the learning rate bit by bit
def learning_schedule(t, t0, t1):
    return t0*1.0 / (t + t1)

## stochastic gradient descent
theta_path_sgd = []
n_iterations = 50
rnd.seed(42)
theta = rnd.randn(2,1)  # random initialization
m = len(X_b)

t0, t1 = 5, 50  # learning schedule hyperparameters
for epoch in range(n_iterations):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index = rnd.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i, t0, t1)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, "b.")
plt.suptitle('Linear Regression with Stochastic Gradient Descent', fontsize=20)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
print("stochastic gradient descent")
plt.show()

# Mini-batch GD
theta_path_mgd = []
n_iterations = 50
minibatch_size = 20
rnd.seed(42)
theta = rnd.randn(2,1)  # random initialization

# define the learning rate parameter
t, t0, t1 = 0, 10, 1000
for epoch in range(n_iterations):
    random_indices = np.random.permutation(m)
    X_b_random = X_b[random_indices]
    y_random = y[random_indices]
    for i in range(0, m, minibatch_size):
	t += 1
	X_i = X_b_random[i: i+minibatch_size]
	y_i = y_random[i: i+minibatch_size]
	gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)
	eta = learning_schedule(t, t0, t1)	
	theta = theta - eta * gradients
	theta_path_mgd.append(theta)

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

## show difference among different GD
plt.figure(figsize=(17,14))
plt.suptitle('Different Gradient Descent Algorithm Convergence Process', fontsize=20)
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()

# cross-validation, split 80/20 train/test set
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import mean_squared_error
def plot_learning_error_curve(theta, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for i in range(1, len(X_train)):
	y_train_predict = X_train[:m].dot(theta)
	y_val_predict = X_val.dot(theta)
	train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
	val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

### Plot Mini-batch GD's error curve
plot_learning_error_curve(theta, X_b, y)
plt.axis([0, 80, 0, 3])
plt.title('Learning Curve of Linear Regression with Mini-batch GD', fontsize=20)
plt.show()
# plot learning curve
# NOTICE: scikit-learn needs ver 0.18 or above
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split 

def plot_learning_curves(model, X, y):
    # automatically split test set & training set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

# learning curve of linear regression
X = 2 * rnd.rand(100, 1)
y = 4 + 3 * X + rnd.randn(100, 1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])
plt.suptitle('Learning Curve of Linear Regression', fontsize=20)
plt.show()

# learning curve for polynomial regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = Pipeline((
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("sgd_reg", LinearRegression()),
    ))

plot_learning_curves(polynomial_regression, X, y)
plt.suptitle('Learning Curve of Polynomial Regression', fontsize=20)
plt.axis([0, 80, 0, 3])
plt.show()





