import os
import csv
import sys
import tensorflow as tf

# Directories for training and tfrecord
Train_Dir="/media/hadoop/5AA07829A0780E2F/EHL/train/"
TFrecord_Dir="/media/hadoop/5AA07829A0780E2F/"

# Image variables
colorspace = 'RGB'
channels = 3
image_format = 'JPEG'

# Getting labels from csv file
reader = csv.DictReader(open('trainLabels.csv'))
mydict = dict((rows['image'],rows['level']) for rows in reader)

# initializing all the variables
init_op = tf.global_variables_initializer()

# making values to list
def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Get all the files in the Directory
matching_files = tf.gfile.Glob(Train_Dir+"*")
output_filename = "Train_TFrecord"
output_file = os.path.join(TFrecord_Dir, output_filename)
writer = tf.python_io.TFRecordWriter(output_file)
fileNum=0
# Place holders for the images
decode_jpeg_data = tf.placeholder(dtype=tf.string)
decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)

# Starting the session
with tf.Session() as sess:
    for file in matching_files:
        print("Processing file: " + str(fileNum))
        fileNum += 1
        # reading each file
        with tf.gfile.FastGFile(file, 'rb') as f:
            image_data = f.read()
        # Decode the RGB JPEG.
        try:
            image = sess.run(decode_jpeg,feed_dict={decode_jpeg_data: image_data})
        except Exception as e:
            print(e)
            print('SKIPPED: Unexpected eror while decoding %s.' % file)
            continue
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        # key for each image
        key = (file.split(".")[0]).split("/")[6]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
            'image/channels': _int64_feature(channels),
            'image/class/label': _int64_feature(int(mydict[key])),
            'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
            'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(file))),
            'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data))}))

        # Serializing and writing to the single file
        writer.write(example.SerializeToString())
        # flush memory
        if (fileNum % 100 == 0):
            sys.stdout.flush()
writer.close()
sys.stdout.flush()
