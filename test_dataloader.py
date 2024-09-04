import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *

import numpy as np

from matplotlib import *
from matplotlib import pyplot
from matplotlib.pyplot import *

from PIL import Image

import glob 
import os
import sys

from skimage import metrics
from skimage.metrics import structural_similarity as ssim


#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#check current directory and import images
print(os.getcwd()) 
filelist=glob.glob('keras_png_slices_train/*.png') # Do not add the 
train_size = len(filelist)
print(train_size)
images=np.array([np.array(Image.open(i),dtype="float32") for i in filelist[0:train_size]])
print('training images',images.shape)