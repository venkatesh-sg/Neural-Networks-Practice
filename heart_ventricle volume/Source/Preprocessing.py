import os
import csv
import sys
import numpy as np
import dicom as pdicom
import tensorflow as tf

# Directories for training and tfrecord
InputDirectory="/media/hadoop/5AA07829A0780E2F/train/"
TFrecord_Dir="/media/hadoop/5AA07829A0780E2F/Heart/"

# Getting labels from csv file
reader = csv.DictReader(open('train.csv'))
Diastole_dict = dict((rows['Id'],rows['Diastole']) for rows in reader)
reader = csv.DictReader(open('train.csv'))
Systole_dict=dict((rows['Id'],rows['Systole']) for rows in reader)



# initializing all the variables
init_op = tf.global_variables_initializer()

# making values to list
def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


patients=os.listdir(InputDirectory)
patients.sort()


output_filename = "Train_TFrecord1"
output_file = os.path.join(TFrecord_Dir, output_filename)
writer = tf.python_io.TFRecordWriter(output_file)

#Place holders for the images
encode_jpeg_data = tf.placeholder(tf.uint16)
encode_jpeg_data=tf.cast(encode_jpeg_data,tf.uint8)
encode_jpeg = tf.image.encode_jpeg(encode_jpeg_data)

fileNum=0

# Starting the session
with tf.Session() as sess:
  for patient in patients:

    print("Processing patient : "+str(patient))
    slices = os.listdir(InputDirectory + patient + "/study/")
    slice_num = []

    for slice in slices:
      if (slice.split("_")[0] == 'sax'):
        slice_num.append(int(slice.split("_")[1]))

    slice_num.sort()
    step_value = (float(Diastole_dict[patient]) - float(Systole_dict[patient])) / (len(slice_num)-1)

    for slice in slices:
      if (slice.split("_")[0] == 'sax'):

        volume=float(Diastole_dict[patient]) - (slice_num.index(int(slice.split("_")[1])) *step_value)
        slices_files=tf.gfile.Glob(InputDirectory+patient+"/study/"+slice+"/*")

        for file in slices_files:
          fileNum+=1
          print("File: "+str(fileNum))
          Filedata = pdicom.read_file(file)
          image_data=Filedata.pixel_array
          image_data=image_data.reshape(Filedata.Rows,Filedata.Columns,1)

          image=sess.run(encode_jpeg,feed_dict={encode_jpeg_data: image_data})

          if(Filedata.PatientsAge[3]=="Y"):
            age=int(Filedata.PatientsAge.split("Y")[0])
          else:
            age=int(round(int(Filedata.PatientsAge.split("M")[0])/12))
          sex=Filedata.PatientsSex

          example = tf.train.Example(features=tf.train.Features(feature={
            'image/volume': _float_feature(volume),
            'image/age': _int64_feature(age),
            'image/sex': _bytes_feature(tf.compat.as_bytes(sex)),
            'image/encoded': _bytes_feature(tf.compat.as_bytes(image))
          }))

          # Serializing and writing to the single file
          writer.write(example.SerializeToString())
          # flush memory
          if (fileNum % 100 == 0):
            sys.stdout.flush()
writer.close()
sys.stdout.flush()