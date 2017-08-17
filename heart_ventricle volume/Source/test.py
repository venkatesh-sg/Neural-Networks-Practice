import tensorflow as tf
import os
import math
import csv
import numpy as np
tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.Session()
import dicom as pdicom

TestDir="Test/"

# Height and width for resizing images
target_height = 200
target_width = 200

# Getting labels from csv file
reader = csv.DictReader(open('train.csv'))
Diastole_dict = dict((rows['Id'],rows['Diastole']) for rows in reader)
reader = csv.DictReader(open('train.csv'))
Systole_dict=dict((rows['Id'],rows['Systole']) for rows in reader)



print("Importing saved model")
new_saver = tf.train.import_meta_graph('Output/model/00000001/export.meta')
new_saver.restore(sess, 'Output/model/00000001/export')

for v in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
    print(v.name)

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("volume_:0")
z = graph.get_tensor_by_name("age:0")
w = graph.get_tensor_by_name("Gender:0")

Squared_error = graph.get_tensor_by_name("mean_squared_error/value:0")

image = tf.placeholder(tf.uint16,[None,None,1])
c_image =tf.cast(image,tf.uint8)
resized_image = tf.image.resize_images(c_image,[target_height,target_width])

TestFiles=tf.gfile.Glob(TestDir+"*/*/*")
Final_Loss=0
for file in TestFiles:
    print(file)
    Filedata = pdicom.read_file(file)
    image_data = Filedata.pixel_array
    image_data = image_data.reshape(Filedata.Rows, Filedata.Columns, 1)

    Image=sess.run(resized_image,feed_dict={image:image_data })

    Sfile=file.split("/")
    if(Sfile[2]=="Diastole"):
        volume=Diastole_dict[Sfile[1]]
    else:
        volume=Systole_dict[Sfile[1]]

    if (Filedata.PatientsAge[3] == "Y"):
        age = int(Filedata.PatientsAge.split("Y")[0])
    else:
        age = int(round(int(Filedata.PatientsAge.split("M")[0]) / 12))

    sex = Filedata.PatientsSex
    if(sex=="M"):
        Gender=20
    else:
        Gender=10

    loss=sess.run(Squared_error, feed_dict={x:[Image],y:[[volume]],z:[[age]],w:[[Gender]]})
    Final_Loss+=loss
    print("loss: "+str(math.sqrt(loss))+"\n")

print("Final loss:"+ str(math.sqrt(Final_Loss/len(TestFiles))))