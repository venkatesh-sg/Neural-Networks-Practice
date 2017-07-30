import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
# Height and width
target_height = 600
target_width = 900

# Directory for test images to visualize
Test_Dir="Test_images/"

# Place holder
x = tf.placeholder(tf.float32, [target_height, target_width, 3], name='x')

# Plotting the output units accirding to number of filters
def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 5
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest")
    plt.show()


# Weights initialization function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

# First convolution and pooling layers with 16 filters of 5*5 size
W_conv1 = weight_variable([5, 5, 3, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, target_height, target_width, 3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolution and pooling layers with 32 filters of 5*5 size
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Third Convolution and pooling layers with 64 filters of size 5*5
W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# Fourth Convolution and pooling layers with 128 filters of size 5*5
W_conv4 = weight_variable([5, 5, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

# Fully connected layer
W_fc1 = weight_variable([3 * 4 * 128, 5])
b_fc1 = bias_variable([5])


h_pool4_flat = tf.reshape(h_pool4, [-1, 3 * 4 * 128])

y_conv = tf.matmul(h_pool4_flat, W_fc1) + b_fc1

# Strarting the session for visualization
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # get all the files in test directory to a single list
    matching_files = tf.gfile.Glob(Test_Dir + "*/*")
    # Iterate through all images
    for image in matching_files:
        # Get labels from path
        Label = image.split("/")[1]
        # Read and decode image data
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        image = tf.image.decode_jpeg(image_data, channels=3)
        resized_image = tf.image.resize_images(images=image, size=[target_height, target_width])
        # Get the units from layer specified
        units = sess.run(h_conv2, feed_dict={x: resized_image.eval()})
        # Plot the units
        plotNNFilter(units)
