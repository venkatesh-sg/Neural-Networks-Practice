from keras.models import Sequential
from keras.layers import Dense, Activation


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Neural Network
model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# Configuring  Optimizer, loss, accuracy
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
(x_train, y_train)=mnist.train.images,mnist.train.labels
model.fit(x_train, y_train,batch_size=32,epochs=16,verbose=1)
(x_test, y_test)=mnist.test.images,mnist.test.labels
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])