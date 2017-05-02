
## Libraries
import random
import numpy as np
import pandas as pd
from time import time

#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import keras.callbacks as kcb

# Global Contrast Normalization
def norm_input(x): return (x-mean_px)/std_px


## Read data from CSV file 
train = pd.read_csv("./convnet_MNIST/train.csv").values
train, train_label = train[:,1:], train[:,0]
train = train.astype('float32')
train_label = train_label.astype('int8')
split_size = int(train.shape[0]*0.85)
train_x, val_x = train[:split_size, :], train[split_size:, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.int8), (np.arange(10)==val_y).astype(np.int8)
mean_px, std_px = train.mean(), train.std()
del train, train_label


## Data augmentation
tr_expand_x = np.zeros((4*train_x.shape[0], train_x.shape[1]), dtype='float32')
tr_expand_y = np.zeros((4*train_y.shape[0], train_y.shape[1]), dtype='int8')
i, j = 0, 0
for x, y in zip(train_x, train_y):
	image = np.reshape(x, (-1, 28))
	j += 1
	if j % 1000 == 0: print "Expanding image number", j
	# iterate over data telling us the details of how to
	# do the displacement
	for d, axis in [(2, 0), (-2, 0), (2, 1), (-2, 1)]:
		new_im = np.roll(image, d, axis)
		tr_expand_x[i], tr_expand_y[i] = new_im.reshape(784), y
		i += 1

train_x = np.vstack([train_x, tr_expand_x])
train_y = np.vstack([train_y, tr_expand_y])


## Create model
hidden_num_units = 512
label_units = 10

model = Sequential([
	Lambda(norm_input, input_shape=(1,28,28), output_shape=(1,28,28)),
    Convolution2D(32,3,3, activation='relu'),
    BatchNormalization(axis=1),
    Convolution2D(32,3,3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(axis=1),
    Convolution2D(64,3,3, activation='relu'),
    BatchNormalization(axis=1),
    Convolution2D(64,3,3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
	Dense(hidden_num_units, init='he_normal'),
	BatchNormalization(),
	Activation('relu'),
	Dropout(0.5),
	Dense(hidden_num_units, init='he_normal'),
	BatchNormalization(),
	Activation('relu'),
	Dropout(0.5),
	Dense(label_units, activation='softmax')
	])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


## Training!!
# make a callback for model checkpoint
class CallMetric(kcb.Callback):
    def on_train_begin(self, logs={}):
        self.best_acc = 0.0
        self.accs = []
        self.val_accs = []
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
    	self.accs.append(logs.get('acc'))
    	self.val_accs.append(logs.get('val_acc'))
    	self.losses.append(logs.get('loss'))
    	self.val_losses.append(logs.get('val_loss'))
    	if logs.get('val_acc') > self.best_acc:
    		self.best_acc = logs.get('val_acc')
    		print "\nThe BEST val_acc to date."


epochs = 12
batch_size = 64

print "Start training..."
t0 = time()
metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath="./cnn_6l.h5", monitor='val_acc', verbose=1, save_best_only=True)
trained_model = model.fit(train_x.reshape(-1, 1, 28, 28), train_y, nb_epoch=epochs, 
	batch_size=batch_size, validation_data=(val_x.reshape(-1, 1, 28, 28), val_y),
	callbacks=[metricRecords, checkpointer])
print "\nElapsed time:", time()-t0, 'seconds\n\n'


model.load_weights('./cnn_6l.h5')
randSample = random.sample(np.arange(train_dataset.shape[0]), 5000)
pred_tr = model.predict_classes(train_x[randSample].reshape(-1, 1, 28, 28))
print "training accuracy:", np.mean(pred_tr==np.argmax(train_y, 1))

pred_val = model.predict_classes(val_x.reshape(-1, 1, 28, 28))
print "validation accuracy:", np.mean(pred_val==np.argmax(val_y, 1))

# Save performance data figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(np.arange(epochs)+1, metricRecords.accs, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_accs, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('cnn_6l_acc.png')

plt.plot(np.arange(epochs)+1, metricRecords.losses, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_losses, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Log loss')
plt.legend(loc='lower right')
plt.savefig('cnn_6l_loss.png')


## Test evaluation
test = pd.read_csv("./convnet_MNIST/test.csv").values
test = test.astype('float32')
pred = model.predict_classes(test.reshape(-1, 1, 28, 28))

# export Kaggle submission file
fh = open('submit.txt','w+')
fh.write('ImageId,Label\n')
for i in xrange(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))

fh.close()


