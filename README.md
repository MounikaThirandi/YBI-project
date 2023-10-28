
Import Library

import pandas as pd

     

import numpy as np
     

import matplotlib.pyplot as plt

     
Import data

from sklearn.datasets import load_digits
     

df= load_digits()
     

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
for ax, image, label in zip(axes, df.images, df.target):
  ax.set_axis_off()
  ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
  ax.set_title("Training: %i" % label)
     

Data Preprocessing

df.images.shape
     
(1797, 8, 8)

df.images[0]
     
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])

df.images[0].shape
     
(8, 8)

len(df.images)
     
1797

n_samples=len(df.images)
data = df.images.reshape((n_samples,-1))
     

data[0]
     
array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])

data[0].shape
     
(64,)

data.shape
     
(1797, 64)
Scaling Image Data

data.min()
     
0.0

data.max()
     
16.0

data=data/16
     

data.min()
     
0.0

data.max()
     
1.0

data[0]
     
array([0.    , 0.    , 0.3125, 0.8125, 0.5625, 0.0625, 0.    , 0.    ,
       0.    , 0.    , 0.8125, 0.9375, 0.625 , 0.9375, 0.3125, 0.    ,
       0.    , 0.1875, 0.9375, 0.125 , 0.    , 0.6875, 0.5   , 0.    ,
       0.    , 0.25  , 0.75  , 0.    , 0.    , 0.5   , 0.5   , 0.    ,
       0.    , 0.3125, 0.5   , 0.    , 0.    , 0.5625, 0.5   , 0.    ,
       0.    , 0.25  , 0.6875, 0.    , 0.0625, 0.75  , 0.4375, 0.    ,
       0.    , 0.125 , 0.875 , 0.3125, 0.625 , 0.75  , 0.    , 0.    ,
       0.    , 0.    , 0.375 , 0.8125, 0.625 , 0.    , 0.    , 0.    ])
Train Test Split Data

from sklearn.model_selection import train_test_split
     

X_train, X_test, y_train, y_test = train_test_split(data, df.target, test_size=0.3)
     

X_train.shape, X_test.shape, y_train.shape, y_test.shape
     
((1257, 64), (540, 64), (1257,), (540,))
Random Forest Model

from sklearn.ensemble import RandomForestClassifier
     

rf= RandomForestClassifier()
     

rf.fit(X_train, y_train)
     
RandomForestClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Predict Test Data

y_pred=rf.predict(X_test)
     

y_pred
     
array([7, 5, 8, 8, 2, 5, 5, 1, 2, 1, 6, 7, 5, 9, 6, 8, 8, 5, 4, 5, 1, 2,
       9, 8, 3, 3, 1, 3, 5, 4, 8, 1, 9, 6, 3, 8, 4, 4, 0, 1, 0, 0, 1, 1,
       4, 6, 3, 7, 5, 9, 5, 4, 3, 4, 0, 2, 2, 7, 4, 7, 6, 6, 5, 4, 9, 2,
       1, 8, 2, 0, 1, 4, 0, 2, 7, 1, 7, 5, 7, 9, 8, 1, 5, 2, 8, 7, 0, 4,
       3, 6, 9, 5, 6, 3, 3, 3, 6, 0, 1, 8, 1, 8, 8, 9, 0, 4, 9, 8, 7, 0,
       3, 5, 6, 4, 8, 2, 4, 3, 1, 6, 0, 0, 2, 5, 6, 1, 9, 0, 8, 8, 4, 7,
       6, 8, 6, 7, 6, 7, 0, 4, 9, 9, 9, 2, 5, 4, 8, 8, 0, 2, 4, 1, 3, 1,
       0, 8, 4, 2, 3, 6, 5, 1, 0, 1, 5, 4, 0, 5, 9, 0, 9, 9, 4, 6, 8, 2,
       2, 8, 1, 9, 1, 0, 9, 9, 8, 4, 2, 7, 6, 0, 1, 1, 9, 0, 0, 9, 8, 0,
       1, 4, 2, 7, 7, 5, 6, 3, 8, 2, 8, 5, 5, 7, 1, 5, 7, 6, 0, 3, 2, 2,
       8, 4, 5, 4, 4, 6, 5, 9, 0, 8, 4, 1, 7, 7, 3, 4, 5, 1, 3, 2, 3, 8,
       5, 9, 7, 6, 7, 3, 5, 2, 8, 0, 5, 6, 9, 1, 4, 5, 2, 6, 4, 3, 0, 8,
       6, 9, 5, 9, 8, 4, 7, 2, 0, 4, 2, 9, 6, 9, 0, 7, 0, 6, 9, 7, 7, 2,
       4, 1, 9, 4, 3, 9, 7, 5, 8, 7, 8, 6, 3, 2, 6, 1, 6, 5, 7, 4, 5, 0,
       4, 2, 5, 6, 9, 6, 6, 0, 4, 0, 5, 2, 0, 3, 9, 2, 6, 6, 5, 7, 3, 2,
       9, 7, 5, 4, 2, 7, 1, 7, 9, 0, 7, 6, 4, 0, 1, 4, 1, 1, 9, 7, 8, 9,
       7, 9, 3, 0, 0, 2, 9, 8, 0, 8, 9, 6, 6, 3, 8, 3, 0, 1, 5, 8, 3, 8,
       4, 1, 4, 6, 5, 8, 3, 0, 7, 7, 4, 1, 5, 6, 3, 3, 1, 7, 6, 7, 3, 9,
       9, 4, 5, 3, 4, 4, 9, 3, 0, 5, 9, 7, 2, 2, 5, 1, 1, 6, 9, 8, 5, 8,
       3, 4, 5, 4, 2, 2, 6, 2, 6, 6, 9, 4, 4, 5, 8, 3, 1, 9, 9, 1, 7, 6,
       6, 9, 7, 9, 4, 7, 4, 3, 0, 0, 6, 1, 5, 8, 7, 9, 1, 5, 9, 5, 4, 8,
       2, 4, 0, 4, 5, 8, 4, 5, 7, 7, 5, 5, 6, 9, 3, 6, 5, 8, 6, 3, 0, 6,
       7, 6, 7, 5, 7, 4, 1, 2, 9, 2, 4, 4, 4, 9, 8, 8, 8, 0, 5, 3, 3, 6,
       1, 7, 2, 6, 2, 1, 3, 9, 7, 3, 9, 7, 0, 1, 5, 1, 5, 5, 6, 3, 7, 6,
       3, 8, 0, 4, 0, 2, 1, 2, 0, 3, 3, 4])
Model Accuracy

from sklearn.metrics import confusion_matrix, classification_report
     

confusion_matrix(y_test, y_pred)
     
array([[52,  0,  0,  0,  1,  0,  0,  0,  0,  0],
       [ 0, 49,  0,  0,  0,  1,  0,  0,  0,  0],
       [ 0,  0, 46,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 48,  0,  1,  0,  0,  1,  0],
       [ 0,  0,  0,  0, 61,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0, 57,  0,  0,  0,  1],
       [ 0,  0,  0,  0,  0,  0, 57,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 52,  0,  1],
       [ 0,  2,  0,  0,  0,  0,  0,  0, 51,  0],
       [ 0,  0,  0,  0,  0,  1,  0,  2,  1, 55]])

print(classification_report(y_test, y_pred))
     
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        53
           1       0.96      0.98      0.97        50
           2       1.00      1.00      1.00        46
           3       1.00      0.96      0.98        50
           4       0.98      1.00      0.99        61
           5       0.95      0.98      0.97        58
           6       1.00      1.00      1.00        57
           7       0.96      0.98      0.97        53
           8       0.96      0.96      0.96        53
           9       0.96      0.93      0.95        59

    accuracy                           0.98       540
   macro avg       0.98      0.98      0.98       540
weighted avg       0.98      0.98      0.98       540
