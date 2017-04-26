from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("/home/hadoop/code/tensorflow_tutorials/python/MNIST_data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

features = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
DNN_classifier = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=features)
DNN_classifier.fit(x=X_train, y=y_train, batch_size=50, steps=1000)

from sklearn.metrics import accuracy_score
y_predict = list(DNN_classifier.predict(X_test))
accuracy = accuracy_score(y_test, y_predict)
print("-------------------- Accuracy: ", accuracy)

from sklearn.metrics import log_loss
y_pred_proba = list(DNN_classifier.predict_proba(X_test))
print("----------------------- Log loss: ", log_loss(y_test, y_pred_proba))
