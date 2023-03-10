# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:02:21 2022

@author: Alfred
"""

from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import RepeatVector, TimeDistributed
import sklearn.metrics as skm
import math
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from tensorflow import keras 
import tensorflow as tf
import time
import os
from sklearn.metrics import r2_score
from numpy import sqrt   
from sklearn.metrics import mean_squared_error,mean_absolute_error
import joblib
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:

    print(e)

#%%


#  X_train
data1 = joblib.load(os.path.join(r''))
x_train=data1




#  x_test
data2 = joblib.load(os.path.join(r''))
x_test=data2



# #  Y_train
data3 = joblib.load(os.path.join(r''))
y_train = data3



# #  y_test
data4 = joblib.load(os.path.join(r''))
y_test = data4




x_train = data1.reshape()

x_test = data2.reshape()




# In[7]:
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D,Activation
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K
from keras.callbacks import LearningRateScheduler
 



def binary_focal_loss(gamma=3, alpha=0.20):

    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed






model = Sequential()

model.add(Conv2D(16, input_shape=(10,15,15), kernel_size=(3,3),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), ))

model.add(Conv2D(32, kernel_size=(3,3),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(1,1),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1), ))


model.add(Conv2D(64, kernel_size=(3,3), padding="same",data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(1,1),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), ))


model.add(Conv2D(128, kernel_size=(3,3), padding="same",data_format='channels_first'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1), ))



model.add(Flatten())
model.add(Dense(64, activation = "relu" ))
model.add(Dense(32, activation = "relu"))
model.add(Dense(1, activation = "sigmoid")) 




filepath=(r'')
checkpoint =keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True
                                , mode='max', period=1, save_weights_only=False)





reduce_lr =  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,verbose=0, patience=20, mode='auto', min_delta=0.00001)
# reduce_lr = LearningRateScheduler(scheduler)


print(model.summary())   



adam = Adam(lr=0.1, decay=0.0,beta_1=0.9,beta_2=0.999,amsgrad=True)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])    #    binary_crossentropy   loss=[binary_focal_loss(alpha=.25, gamma=2)]
history = model.fit(x_train, y_train,validation_data=(x_test,y_test),shuffle=True, batch_size=64, epochs=1000, callbacks=[checkpoint,reduce_lr],verbose=1)




history_dict = history.history
history_dict.keys()


model.save(r'')
model = tf.keras.models.load_model('')
model.load_weights(r'')

# In[10]:


# ??????????????????????????????????????????

import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


plt.plot(loss_values, 'b',color = 'blue', label=' loss')
plt.plot(val_loss_values, 'b',color='red', label='val_loss')
plt.rc('font', size = 18)
plt.title('Training accuracy and  loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(15,7)
fig.savefig('img/tcstest&validationlosscnn.png', dpi=300)
plt.show()


# In[11]:



mae = history_dict['val_accuracy']
vmae = history_dict['accuracy']

plt.plot( mae, 'b',color = 'red', label='val_accuracy')
plt.plot( vmae, 'b',color='blue', label='accuracy')
plt.title('validation accuracy and  error')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(15,7)
fig.savefig('img/tcstest&validationerrorcnn.png', dpi=300)
plt.show()


# In[12]:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns



model.metrics_names


# # In[13]:


trainScore = model.evaluate(x_train, y_train, verbose=1)
print("train loss, train acc:", trainScore)

testScore = model.evaluate(x_test, y_test, verbose=1)
print("test loss, test acc:", testScore)


#%%
#  ????????????(??????+???????????????)????????????
predict = joblib.load(os.path.join(r''))
#  ??????????????????(label)
anslabel = pd.read_csv(r'',header=None)   



predict = predict.reshape(4218,10,15,15)
p = model.predict(predict)
y_pred = p




cm = confusion_matrix(anslabel,np.round(abs(y_pred)),labels=[0,1])

df_cm = pd.DataFrame(cm)

ax = sns.heatmap(df_cm,annot=True,fmt='.20g')
ax.set_title('confusion matrix') 
ax.set_xlabel('predict') 
ax.set_ylabel('true') 


#%%
# Accuracy
accuracy=accuracy_score(anslabel   ,np.round(abs(y_pred)))

# Precision
precision=precision_score(anslabel   ,np.round(abs(y_pred)))

# recall_score
recall=recall_score(anslabel   ,np.round(abs(y_pred)))

# f1_score
f1_score=f1_score(anslabel   ,np.round(abs(y_pred)))

print(accuracy)
print(precision)
print(recall)
print(f1_score)



#%%

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

 

fpr,tpr,threshold = roc_curve(anslabel, y_pred) 
roc_auc = auc(fpr,tpr)
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

