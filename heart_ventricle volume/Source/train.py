import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import time
# staring time of training
start = int(round(time.time() * 1000))

# Directory for tfrecord file, exporting the model and logs
Train_Dir = '/media/hadoop/5AA07829A0780E2F/Heart/Train_TFrecord1'
export_path = 'Output/model'
log_path='Output/logs'

# Height and width for resizing images
target_height = 200
target_width = 200

# Reading and decoding from the TFrecord file
def read_and_decode(filename_queue):
    # Reader
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/volume': tf.FixedLenFeature([], tf.float32),
            'image/age': tf.FixedLenFeature([], tf.int64),
            'image/sex': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)
        }
    )

    # Getting the encoded binary image and decode to jpeg
    image = tf.image.decode_jpeg(features['image/encoded'],channels=1)
    # Resize to 200*200 images to reduce training time
    resized_image = tf.image.resize_images(images=image,size=[target_height,target_width])
    # Get the blood volume for the image
    volume=[features['image/volume']]
    tf.Print(volume,data=[volume])
    Age=[features['image/age']]
    Sex=[features['image/sex']]
    # Shuffle the images and batch them to train
    images, volumes,sex,age = tf.train.shuffle_batch([resized_image, volume, Sex, Age],
                                                 batch_size=20,
                                                 capacity=50,
                                                 num_threads=5,
                                                 min_after_dequeue=20)

    return images, volumes, sex, age

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
# Start the FIFO Queue for getting images constantly
filename_q = tf.train.string_input_producer([Train_Dir],num_epochs=5000)
# images and labels for images in queue
images, volumes, sex, age = read_and_decode(filename_q)

# Place holders for inputs of the deep network
x = tf.placeholder(tf.float32, [None, target_height, target_width, 1], name='x')
y_ = tf.placeholder(tf.float32, [None, 1], name='volume_')
z =tf.placeholder(tf.float32,[None, 1],name='age')
w=tf.placeholder(tf.float32,[None, 1],name='Gender')

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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First convolution and pooling layers with 16 filters of 5*5 size
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, target_height, target_width, 1])
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

#tf.Print(h_pool4,data=[h_pool4.shape])
# Fully connected layer
W_fc1 = weight_variable([13 * 13 * 128, 10])
b_fc1 = bias_variable([10])
h_pool4_flat = tf.reshape(h_pool4, [-1, 13 * 13 * 128])

y_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
y_fc1c=tf.concat([y_fc1,z,w],1)

W_fc2 = weight_variable([12, 5])
b_fc2 = bias_variable([5])
y_fc2 = tf.nn.relu(tf.matmul(y_fc1c, W_fc2) + b_fc2)

W_fc3 = weight_variable([5, 1])
b_fc3 = bias_variable([1])
y_fc3 =tf.nn.relu(tf.matmul(y_fc2, W_fc3)+b_fc3)

MSE=tf.losses.mean_squared_error(y_fc3,y_)

# training step to back propagate the cross entropy error
train_step = tf.train.AdamOptimizer(0.0001).minimize(MSE)

# Summarizing to visualize using Tensorboard later
tf.summary.scalar('squared error', MSE)
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

    # Iterating over 5000 iteration to train the model
    for i in range(10000):
        TrainImages, TrainVolume, TrainSex, TrainAge = sess.run([images, volumes, sex, age])
        TrainGender=[]
        for entry in TrainSex:
            if(entry[0]=="F"):
                TrainGender.append([10])
            else:
                TrainGender.append([20])
        summary, Error, training = sess.run([merged, MSE, train_step], feed_dict={x: TrainImages, y_: TrainVolume, z: TrainAge, w: TrainGender})
        print("Iteration:" + str(i) + " Batch Error: " + str(Error))
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
            'volumes': exporter.generic_signature({'volumes': y_})})

    model_exporter.export(export_path, tf.constant(1), sess)

    # Ending time of training
    end = int(round(time.time() * 1000))
    print("Time for building model: ")
    print(end - start)
