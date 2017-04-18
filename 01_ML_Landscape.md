## My Answers to exercises in Chapter 1 in BOOK 1 (references)
### 1. How would you define Machine Learning?
> Machine Learning is about making machines get better performance at some tasks by learning from data without being explicitly programmed

### 2. Can you name four types of problems where it shines?
> Machine Translation, Image Recognition, Speech Recognition, Go game, Spam filter, etc.

>     **Refrenced Answer**: Fit for 4 types of problems:

* Complex problem for which there is no good traditional algorithmic approach to solve it
* Existing solutions requires long lists of hardcoded rules or hand-tuning
* Fluctuating environment which is difficult for traditional solutions to adapt to.
* Unclear pattern in huge amount of data

### 3. What is a labeled training set?
> Desired result for each instance in training set. it's our formalized understanding about the data, used for training & improving the effectiveness of learning algorithm

### 4. What are the two most common supervised tasks?
>  classification & regression

### 5. Can you name four common unsupervised tasks?
> clustering & dimension reducing & visualization & association rule discovery(learning)

### 6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?
> reinforcement learning, an agent learns to act according to the best learned policy by maxmizing the rewards. especially in a unknown environment, improve the performance by trial & errors

### 7. What type of algorithm would you use to segment your customers into multiple groups?
> clustering if no standard provided and no group number required otherwise classification

### 8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
> supervised (using labeled spam email to train the model)

### 9. What is an online learning system?
> incrementally learn the model on the fly. As opposed to batch learning, it makes the model adapt easily to the changing data and autonomous system, and it can handle large amount of data.

### 10. What is out-of-core learning?
> online learning, when the training data set is too large to fit in the memory. An online learning model chops the data into small chunk, i.e. mini-batch, and learning from the mini-bacth to incrementally improve the model.

### 11. What type of learning algorithm relies on a similarity measure to make predictions?
> instance-based learning. (learn by heart), find the similar instance to make predictions.

### 12. What is the difference between a model parameter and a learning algorithm’s hyperparameter?
> hyperparameters are initialized prior to the training and remain unchanged in the process, while model' parameters are adjusted along the training process to get optimal ones

> So, to summarize. [Hyperparameters:](https://www.quora.com/Machine-Learning/What-are-hyperparameters-in-machine-learning)
* Define higher level concepts about the model such as complexity, or capacity to learn.
* Cannot be learned directly from the data in the standard model training process and need to be predefined.
* Can be decided by setting different values, training different models, and choosing the values that test better

Some examples of hyperparameters:
* Learning rate (in many models)
* Number of hidden layers in a deep neural network
* Number of clusters in a k-means clustering
* Number of leaves or depth of a tree
* Number of latent factors in a matrix factorization

### 13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
> optimal value for model parameters, minimize the loss function (and penalty for model complexity if regularized)

### 14. Can you name four of the main challenges in Machine Learning?
> data not enough, nonrepresentative data, poor quality data (need denoising, fill in the missing value), irrelevant features, overfitting (too complex), underfitting (too simple)

### 15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
> might be overfitting which has lower generalization ability;
* get more training data;
* make the model simpler (regularization, simpler algorithms/parameters/features)
* denoise (remove errors, outliers)

### 16. What is a test set and why would you want to use it?
> a unused labeled dataset for generalization capability examination

### 17. What is the purpose of a validation set?
> a second hold out of data to validate the effectiveness (generalization capability) of the model, in case that model adapts to the test set, using validate set to select the best model & tune hyperparameters, and give a final examination on test case

### 18. What can go wrong if you tune hyperparameters using the test set?
>  model adapts to the test set which may have a lower generalization ability

### 19. What is cross-validation and why would you prefer it to a validation set?
> split the training data into complementary sets, one for training  & the other for validation, it saves training data since it doesn't need a separate validation set for model selection & hyperparameter tuning


