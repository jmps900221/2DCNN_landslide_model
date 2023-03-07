# -*- coding: utf-8 -*-
"""

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
    
    
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


#  X_train
data1 = joblib.load(os.path.join(r''))
x_train = data1

#  x_test
data2 = joblib.load(os.path.join(r''))
x_test = data2

# #  Y_train
data3 = joblib.load(os.path.join(r''))
y_train = data3

# #  y_test
data4 = joblib.load(os.path.join(r''))
y_test = data4


#  Reshape the input data if needed
# x_train = x_train.reshape()
# x_test = x_test.reshape()


#  SVM model
svm = SVC(kernel='rbf', C=1, gamma=0.1)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

# MLP model
mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000)
mlp.fit(x_train, y_train)
y_pred_mlp = mlp.predict(x_test)

# XGBoost model
xgb = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=1000, objective='binary:logistic')
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

#  Evaluation metrics for SVM model
print('SVM model:')
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred_svm))
print('Accuracy: ', accuracy_score(y_test, y_pred_svm))
print('Precision: ', precision_score(y_test, y_pred_svm))
print('Recall: ', recall_score(y_test, y_pred_svm))
print('F1 score: ', f1_score(y_test, y_pred_svm))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_svm)
roc_auc = auc(fpr, tpr)
print('ROC AUC: ', roc_auc)

#  Evaluation metrics for MLP model
print('MLP model:')
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred_mlp))
print('Accuracy: ', accuracy_score(y_test, y_pred_mlp))
print('Precision: ', precision_score(y_test, y_pred_mlp))
print('Recall: ', recall_score(y_test, y_pred_mlp))
print('F1 score: ', f1_score(y_test, y_pred_mlp))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_mlp)
roc_auc = auc(fpr, tpr)
print('ROC AUC: ', roc_auc)

#  Evaluation metrics for XGBoost model
print('XGBoost model:')
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred_xgb))
print('Accuracy: ', accuracy_score(y_test, y_pred_xgb))
print('Precision: ', precision_score(y_test, y_pred_xgb))
print('Recall: ', recall_score(y_test, y_pred_x))
