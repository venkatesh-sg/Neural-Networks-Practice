import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import time
# staring time of training
start = int(round(time.time() * 1000))

# Directory for tfrecord file, exporting the model and logs
Train_Dir = '/media/hadoop/5AA07829A0780E2F/EHL/Train_TFrecord.tfrecords'
export_path = 'Output/model'
log_path='Output/logs'

# Height and width for resizing images
target_height = 600
target_width = 900

# Reading and decoding from the TFrecord file
def read_and_decode(filename_queue):
    # Reader
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)
        }
    )

    # Getting the encoded binary image and decode to jpeg
    image = tf.image.decode_jpeg(features['image/encoded'],channels=3)
    # Resize to 600*900 images to reduce training time
    resized_image = tf.image.resize_images(images=image,size=[target_height,target_width])
    # Get the labels for the image
    label=tf.one_hot(features['image/class/label'],depth=5)
    # Shuffle the images and batch them to train
    images, labels = tf.train.shuffle_batch([resized_image, label],
                                                 batch_size=10,
                                                 capacity=50,
                                                 num_threads=5,
                                                 min_after_dequeue=20)

    return images, labels

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
# Start the FIFO Queue for getting images constantly
filename_q = tf.train.string_input_producer([Train_Dir],num_epochs=5000)
# images and labels for images in queue
Iimages, Ilabels = read_and_decode(filename_q)

# Place holders for inputs of the deep network
x = tf.placeholder(tf.float32, [None, target_height, target_width, 3], name='x')
y_ = tf.placeholder(tf.float32, [None, 5], name='y_')

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

# softmax layer for classification of images
y_conv = (tf.matmul(h_pool4_flat, W_fc1) + b_fc1,"final_result:0")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# training step to back propagate the cross entropy error
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
# Getting correct prediction for cluculating accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Summarizing to visualize using Tensorboard later
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('Batch Accuracy', accuracy)
merged = tf.summary.merge_all()

# writing the summaries to logs
trainwriter = tf.summary.FileWriter(log_path)

# starting the tensorflow session
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    trainwriter.add_graph(sess.graph)
    # Start populating the filename queue.
    # coordinators for threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    Total_Accu=0
    # Iterating over 5000 iteration to train the model
    for i in range(5000):
        TrainImages, TrainLabels = sess.run([Iimages, Ilabels])
        summary, accu, training = sess.run([merged,accuracy,train_step], feed_dict={x: TrainImages, y_: TrainLabels})
        Total_Accu += accu
        print("Iteration:" + str(i)+" Batch Accuracy: " + str(accu)+" Total Accuracy: " + str(Total_Accu/(i+1)))
        trainwriter.add_summary(summary, i)
    coord.request_stop()
    coord.join(threads)


    # Saving the model forture testing
    print('Exporting trained model to', export_path)
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'images': x}),
            'outputs': exporter.generic_signature({'scores': y_})})

    model_exporter.export(export_path, tf.constant(1), sess)

    # Ending time of training
    end = int(round(time.time() * 1000))
    print("Time for building convnet: ")
    print(end - start)
