import tensorflow as tf
sess = tf.InteractiveSession()


#importing MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#Building full computational Graph
#Inputs and Outputs
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#weights and Baises
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))



#predicted output

#using softamx function
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#using softmax with logits
# y = tf.matmul(x,W) + b
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


#training=tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
#training=tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
#training=tf.train.AdagradOptimizer(0.2).minimize(cross_entropy)
#training=tf.train.AdadeltaOptimizer(20).minimize(cross_entropy)
#training=tf.train.MomentumOptimizer(0.1,0.1).minimize(cross_entropy)
training=tf.train.RMSPropOptimizer(0.01).minimize(cross_entropy)


#initializing variables
#Must initialize after optimiser because some optimisers like AdamOptimiser create variables
sess.run(tf.global_variables_initializer())

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(training, feed_dict={x: batch_xs, y_: batch_ys})

#evaluation
#comparing indeces maximum element of both arrays
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
