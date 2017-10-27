Diabetic retinopath 
DR is a disease when the retina of the eye is damaged due to diabetes. It is
the leading cause of blindness in the working-age population of the developed
world. It is estimated to affect over 93 million people. 

The
US Center for Disease Control and Prevention estimates that 29.1 million people
in the US
have diabetes and the World Health Organization estimates that 347 million
people have the disease worldwide. Diabetic Retinopathy (DR) is an eye disease
associated with long-standing diabetes. Around 40% to 45% of Americans with
diabetes have some stage of the disease. Progression to vision impairment can
be slowed or averted if DR is detected in time; however this can be difficult
as the disease often shows few symptoms until it is too late to provide
effective treatment.

The
goal of this project is to develop an automated detection system of Diabetic
retinopathy using neural networks which will be trained on pre classified
images of Diabetic retinopathy.


The Dataset:

Dataset is taken from the kaggle competition
which has a large set of high-resolution retina images taken under a variety of
imaging conditions. It has 35126 JPEG images for training (32.5GB) which are
divided into 5 levels: from 0 to 4, where 0 corresponds to the healthy
state and 4 is the most severe state. Different eyes of the same person can be
at different levels.

The images in the dataset come from different
models and types of cameras, which can affect the visual appearance of left vs.
right. Some images are shown as one would see the retina anatomically
(macula on the left, optic nerve on the right for the right eye). Others are
shown as one would see through a microscope condensing lens (i.e.
inverted, as one sees in a typical live eye exam). 

For comparison between the model trained dataset
is divided into two parts, one for training the neural network and other for
validating the trained model. Training part has 26718 images which are about
76% of total and rest is for validation. Validation of the model has to be done
on the data that neural network is not trained on, so validation data is not
used for training.

