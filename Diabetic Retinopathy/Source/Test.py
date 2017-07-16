import tensorflow as tf
import sys
import numpy as np

# Directory for testing images
Test_Dir="Test_images/"
# Labels collected from labels.txt
labels = [line.rstrip() for line
                   in tf.gfile.GFile("Output/model/output_labels.txt")]

# Reloading the trained model for testing
with tf.gfile.FastGFile("Output/model/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Starting the tensorflow session for testing
with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # Get the softamx tensor for images
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # Get all the images under testing directory
    matching_files = tf.gfile.Glob(Test_Dir+"*/*")
    # counting correct predictions
    correct_predictions=0
    Final_predictions=[]
    Y_Label=[]
    Y_Pred=[]
    # For each image in the test directory
    for image in matching_files:
        # Reading the images
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        # Get prediction for image
        predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
        # Top prediction according to confidence scores
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        # Getting the labels from path
        Label=image.split("/")[1]
        print("Predicted "+Label+" as "+labels[top_k[0]])
        if(labels[top_k[0]]==Label):
            correct_predictions += 1
        Y_Label.append(labels.index(Label))
        Y_Pred.append(top_k[0])
        Final_predictions.append(predictions[0])

    # Saving predictions, True labels and confidence scores to use later in visualization
    np.savetxt("visualization/Predicted.txt",Y_Pred)
    np.savetxt("visualization/TrueLabels.txt",Y_Label)
    np.savetxt("visualization/Confidence_score.txt",Final_predictions)
    print("Accuracy: "+str(correct_predictions/len(matching_files)))