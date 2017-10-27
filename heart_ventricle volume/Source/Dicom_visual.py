import os
import pylab
import dicom as pdicom
import tensorflow as tf


InputDirectory="/media/hadoop/5AA07829A0780E2F/train/"
patients=os.listdir(InputDirectory)
patients.sort()

def scanfiles(path):
    matching_files = tf.gfile.Glob(path+ "/*/*/*")
    return matching_files

for patient in patients:
    patient_scans=scanfiles(InputDirectory+patient)
    print("Numbers of scans for patient "+patient+" are "+str(len(patient_scans)))
    for i in range(len(patient_scans)):
        if i % 30 == 0:
            Filedata = pdicom.read_file(patient_scans[i])
            print("patient sex: "+Filedata.PatientsSex)
            print("patient age: "+Filedata.PatientsAge)
            print("File: "+patient_scans[i])
            pylab.imshow(Filedata.pixel_array, cmap=pylab.cm.bone)
            pylab.show()