
## Libraries
#import os
import numpy as np
import pandas as pd
from time import time
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import keras.callbacks as kcb


## Read data from CSV file 
train = pd.read_csv("../train.csv").values
train, train_label = train[:,1:]/255., train[:,0]
train = train.astype('float32')
split_size = int(train.shape[0]*0.85)
train_x, val_x = train[:split_size, :], train[split_size:, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.float32), (np.arange(10)==val_y).astype(np.float32)
del train, train_label

# data augmentation
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

# define vars
input_num_units = 784
hidden_num_units = 512
label_units = 10

# create model
model = Sequential([
  Dense(hidden_num_units, init='he_normal', input_dim=input_num_units),
  BatchNormalization(),
  Activation('relu'),
  Dropout(0.5),
  Dense(hidden_num_units, init='he_normal'),
  BatchNormalization(),
  Activation('relu'),
  Dropout(0.5),
  Dense(label_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Training!!
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

epochs = 20
batch_size = 64

print "Start training..."
t0 = time()
metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath="./fc_4l_model.h5", monitor='val_acc', verbose=1, save_best_only=True)
trained_model = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, 
  validation_data=(val_x, val_y), callbacks=[metricRecords, checkpointer])
print "\nElapsed time:", time()-t0, 'seconds\n\n'


model.load_weights('./fc_4l_model.h5')
pred_tr = model.predict_classes(train_x)
print "\ntraining accuracy:", np.mean(pred_tr==np.argmax(train_y, 1))

pred_val = model.predict_classes(val_x)
print "\nvalidation accuracy:", np.mean(pred_val==np.argmax(val_y, 1))

# Save performance data figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(np.arange(epochs)+1, metricRecords.accs, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_accs, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('fc_4l_model.png')

plt.plot(np.arange(epochs)+1, metricRecords.losses, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_losses, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Log loss')
plt.legend(loc='lower right')
plt.savefig('fc_4l_model.png')

# Test evaluation
test = pd.read_csv("../test.csv").values
test = test / 255.
pred = model.predict_classes(test)
print "\n"

# export Kaggle submission file
fh = open('eval.txt','w+')
fh.write('ImageId,Label\n')
for i in xrange(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))

fh.close()

# no augmentation
# 416.98 sec, tr_acc=.9982, val_acc=.9806, test_acc=.9793 (layers: 784-1024-10)
# 320.97 sec, tr_acc=.9945, val_acc=.9792, test_acc=.9760 (layers: 784-512-512-10) (25 epochs)
# 398.05 sec, tr_acc=.9952, val_acc=.9800, test_acc=.9781 (layers: 784-512-512-10) (30 epochs)

# with augmentation and batch normalization
# 1104.27 sec, tr_acc=.9977, val_acc=.9898 (layers: 784-1024-10) (20 epochs)
# 1033.15 sec, tr_acc=.9957, val_acc=.9897, test_acc=.9907 (layers: 784-512-512-10) (20 epochs)
# 618.469 sec, tr_acc=.9885, val_acc=.9867, test_acc=.9847 (layers: 784-256-256-256-10) (20 epochs)
# 1369.13 sec, tr_acc=.9959, val_acc=.9916, test_acc=.9896 (layers: 784-512-512-512-10) (20 epochs)
