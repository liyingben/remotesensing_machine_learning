# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 08:39:03 2018
author:lerryw
title:machine learning classification
"""

##Machine Learning=group
##Machine Learning=name
##Image_to_classify=raster
'''###TrainingSet=vector
###Field_ID=field TrainingSet'''
##Training_image=raster
##Machine_Learning_Model=selection RandomForestClassifier;ExtraTreesClassifier;BaggingClassifier;AdaBoostClassifier;DecisionTreeClassifier;GaussianNB;QuadraticDiscriminantAnalysis;KNeighborsClassifier;SVC;MLPClassifier;KMeans
##Classified_image=output raster'

from qgis.core import *
from qgis.gui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from sklearn.ensemble import (RandomForestClassifier, 
                              ExtraTreesClassifier, 
                              BaggingClassifier,
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


# Tell GDAL to throw python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# set data path.
in_img_path = Image_to_classify
train_img_path = Training_image
out_img_path = Classified_image

# import into python
img_ds=gdal.Open(in_img_path, gdal.GA_ReadOnly)
roi_ds=gdal.Open(train_img_path, gdal.GA_ReadOnly)

# convert raster data to array
img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize,img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

# read array for all bands
for b in range (img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

# convert training image into array  
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# subset dataset, X = img dataset, y =  classed from training dataset
X = img[roi > 0, :]
y = roi[roi > 0]

# Mask out clouds, cloud shadows, and snow using Fmask
clear = X[:, 7] <= 1

X = X[clear, :7]  # we can ditch the Fmask band now
y = y[clear]

geo = img_ds.GetGeoTransform()
proj = img_ds.GetProjection()
shape = img.shape

if Machine_Learning_Model==0:
    clf=RandomForestClassifier(n_estimators=1000)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==1:
    clf=ExtraTreesClassifier(n_estimators=1000)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==2:
    clf=BaggingClassifier(n_estimators=1000,)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==3:
    clf=AdaBoostClassifier(n_estimators=1000, learning_rate=1.0)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==4:
    clf=DecisionTreeClassifier(max_depth=10, min_samples_split=2,random_state=0)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==5:
    clf=GaussianNB()
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==6:
    clf=QuadraticDiscriminantAnalysis()
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==7:
    clf=KNeighborsClassifier(3)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==8:
    clf=SVC(kernel="linear", C=0.025)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==9:
    clf=MLPClassifier(alpha=1)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)
    
elif Machine_Learning_Model==10:
    clf=KMeans(n_clusters=5)
    clf.fit(X, y)
    # score = clf.score(X, y)
    new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1) 
    img_as_array = img[:, :, :7].reshape(new_shape)
    # Now predict for each pixel
    class_prediction = clf.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # GeoTIFF output
    outDS = gdal.GetDriverByName('GTiff').Create(Classified_image,shape[1], shape[0], 1, gdal.GDT_Byte)
    outDS.GetRasterBand(1).WriteArray(class_prediction)
    outDS.SetGeoTransform(geo)
    outDS.SetProjection(proj)