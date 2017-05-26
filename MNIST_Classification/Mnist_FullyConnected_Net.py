import random
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()


#importing MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Weights and Biases
def weight(shape):
  initialization = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initialization)

def bias(shape):
  initialization = tf.truncated_normal(shape,0.1)
  return tf.Variable(initialization)

x= tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#first hidden layer
w1=weight([784,100])
b1=bias([100])

y1=tf.nn.relu(tf.matmul(x,w1)+b1)

#Second hidden layer
w2=weight([100,50])
b2=bias([50])

y2=tf.nn.relu(tf.matmul(y1,w2)+b2)

#softmax layer
w3=weight([50,10])
b3=bias([10])

y_logits=tf.matmul(y2,w3)+b3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_logits,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%500 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1],})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))