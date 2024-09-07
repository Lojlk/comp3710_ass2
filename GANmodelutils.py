import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
# from keras.api.models import Sequential
# from keras.api.layers import Dense, LeakyReLU


###defining the generator###
def define_generator():

    model = tf.keras.Sequential()
    
    # Fully connected layer: Takes random input (latent vector of 256 dimensions) and maps it to a larger feature space (32*32*256)
    model.add(Dense(32*32*256, input_shape=(256,)))
    model.add(LeakyReLU(alpha=0.2)) # Activation 
    model.add(Reshape((32, 32, 256)))  # Reshaping into 3D tensor of shape 

    # Upsampling layer: Increases the spatial dimensions from 32x32 to 64x64 while reducing the depth to 256
    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization()) # Normalization to stabilize training 
    model.add(LeakyReLU(alpha=0.2))
    
    # Further upsampling: Goes from 64x64 to 128x128, reducing depth to 128
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    # Final upsampling: From 128x128 to 256x256 while reducing the depth to 64
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    # Further convolution: Keeps the image at 256x256 but refines the depth
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Output layer: Produces the final single-channel image with the same size as the original MRI (256x256x1)
    model.add(Conv2D(1, (3,3),strides=(1,1), padding='same', use_bias=False))
    return model

###defining the discriminator###
def define_discriminator(input=(256,256,1)):

    model = tf.keras.Sequential()
    
    # First convolution: Downscales the image from 256x256 to 128x128 and increases depth to 32
    model.add(Conv2D(32, (4,4), strides=(2, 2), padding='same',input_shape=input))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    # Second convolution: Further downscales to 64x64, increases depth to 64
    model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Third convolution: Downscales to 32x32, increases depth to 128
    model.add(Conv2D(128, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    # Flatten and output: Flattens the 3D output to 1D and 
    # # reduces to a single scalar value (real or fake classification)
    model.add(Flatten())
    model.add(Dense(1))
    return model


