import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import os
import numpy as np
import time
start = int(round(time.time() * 1000))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())


rootdir = 'Data/'
TotalImage=0
TotalLabel=-1

#Counting Number of Images and Classes
print("Counting Number of images and classes")
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        TotalImage=TotalImage+1
    TotalLabel=TotalLabel+1
print("Total Images: "+str(TotalImage))
print("Total Classes: "+str(TotalLabel))


#Getting Labels For each image in to array
print("Getting all classes in to array for future use")
LabelArray = np.ndarray(shape=(TotalLabel), dtype=np.dtype('S20'))
Label=-1 #-1 for main directory Data/
for subdi, dir, files in os.walk(rootdir):
    for fil in files:
        LabelName=subdi.split("/")[1]
        if(Label!=-1):
            LabelArray.itemset(Label,LabelName)
    Label=Label+1


Images = np.ndarray(shape=(TotalImage,10000))
Labels = np.ndarray(shape=(TotalImage,TotalLabel))

print("working on each image")
Image=0
for subdir,dirr,files in os.walk(rootdir):
    for file in files:
        curFile = file
        curLabel = subdir.split("/")[1]


        #training label matrix
        index=np.where(LabelArray==str.encode(curLabel))
        fileLabels = np.zeros(shape=(TotalLabel))
        fileLabels[index[0]] = 1
        Labels[Image] = fileLabels

        file_contents = tf.read_file(os.path.join(subdir, file))
        Imagecontent = tf.image.decode_png(file_contents, channels=1)
        image = tf.image.resize_images(Imagecontent, [28,28])

        with sess.as_default():
            flatImage = image.eval().ravel()
        flatImage = np.multiply(flatImage, 1.0 / 255.0)

        Images[Image]=flatImage







print("counting number of images for testing")
Testdir="TestData/"
TotalImage=0
for subdir, dirsr, files in os.walk(Testdir):
    for file in files:
        TotalImage=TotalImage+1

TestImages = np.ndarray(shape=(TotalImage,784))
TestLabels = np.ndarray(shape=(TotalImage,TotalLabel))


print("working on each image")
Image=0
for subdir,di,files in os.walk(Testdir):
    for file in files:
        curLabel = subdir.split("/")[1]


        #training label matrix
        index=np.where(LabelArray==str.encode(curLabel))
        fileLabels = np.zeros(shape=(TotalLabel))
        fileLabels[index[0]] = 1
        TestLabels[Image] = fileLabels

        file_contents = tf.read_file(os.path.join(subdir, file))
        Imagecontent = tf.image.decode_png(file_contents, channels=1)
        image = tf.image.resize_images(Imagecontent, [28, 28])

        with sess.as_default():
            flatImage = image.eval().ravel()
        flatImage = np.multiply(flatImage, 1.0 / 255.0)

        TestImages[Image]=flatImage



x = tf.placeholder(tf.float32, [None, 784],name='x')
y_ = tf.placeholder(tf.float32, [None,TotalLabel],name='y_')

init = tf.global_variables_initializer()
sess.run(init)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 100, 100, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer
W_fc2 = weight_variable([1024, TotalLabel])
b_fc2 = bias_variable([TotalLabel])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.histogram('Convolution_Weights_1', W_conv1)
tf.summary.histogram('Convolution_1', h_conv1)

tf.summary.histogram('Convolution_Weights_2', W_conv2)
tf.summary.histogram('Convolution_2', h_conv2)

tf.summary.histogram('Keep_prob', keep_prob)
tf.summary.histogram('h_fc1_drop', h_fc1_drop)


tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)

tf.summary.histogram('Accuracy', accuracy)
merged = tf.summary.merge_all()

trainwriter = tf.summary.FileWriter('Output/logs', sess.graph)
sess.run(tf.global_variables_initializer())


for i in range(100):


    mask = np.random.choice([True, False], len(Images), p=[0.15, 0.85])
    trainImages = Images[mask]
    trainLabels = Labels[mask]

    summary, _ = sess.run([merged, train_step], feed_dict={x: trainImages, y_: trainLabels, keep_prob: 0.5})

    trainwriter.add_summary(summary, i)
    if (i % 25 == 0):
        train_accuracy = accuracy.eval(feed_dict={x: trainImages, y_: trainLabels, keep_prob: 1.0},session=sess)
        print("Evaluating iteration %d, training accuracy %g" % (i, train_accuracy))


print("Final Test Accuracy: %g" % accuracy.eval(feed_dict={x: TestImages, y_: TestLabels, keep_prob: 1.0},session= sess))



export_path = 'Output/model'
print('Exporting trained model to', export_path)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'scores': y_})})

model_exporter.export(export_path, tf.constant(1), sess)

end = int(round(time.time() * 1000))
print("Time for building convnet: ")
print(end - start)

