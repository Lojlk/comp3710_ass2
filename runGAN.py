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

# from modelscript import define_generator, define_discriminator
from GANmodelutils import define_generator, define_discriminator


# check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# check current directory and import images
print(os.getcwd()) 

# Use glob to get a list of all PNG image files in the specified directory
filelist=glob.glob('/home/groups/comp3710/OASIS/keras_png_slices_train/*.png')  

# Get the number of images in the dataset
train_size = len(filelist)
print(train_size)

# Load the images and convert them into a NumPy array
# Here, we open each image file, convert it into a NumPy array of float32 type, and store it in a list
# We then convert this list into a 4D NumPy array (batch_size, height, width, channels)
images=np.array([np.array(Image.open(i),dtype="float32") for i in filelist[0:train_size]])

# Check the shape of the loaded images to ensure they are loaded correctly
# This should print the shape as (train_size, height, width) 
print('training images',images.shape) # dimension of each image is 256 x 256

#######################################################################
# Preprocessing
# Normalise pixel values from [0,255] to [-1,1]
# (GANs typically work better when input values are within this range) 
images=(images - 127.5) / 127.5

# Add a new axis to the image array to make it a 4D array (batch_size, height, width, channels)
# The new axis represents the channels, since the images are grayscale, it will have only 1 channel.
images=images[:,:,:,np.newaxis]

# check shape
print(images.shape)

#######################################################################
# Visualize the first 10 brain MRI images from the dataset
pyplot.figure(figsize=(25,25))
for i in range(10):
    # define subplot
    pyplot.subplot(5, 5, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(images[i,:,:,0],cmap="gray")

pyplot.show()
pyplot.savefig("First_10_dataset.png")

########################################################################
# Call the generator and discriminator models

g_model = define_generator() # Initialize the generator model

d_model = define_discriminator() # Initialize the discriminator model

######## Visualising generated images############
#choose the number of samples to visualise
n_samples=5
#define number of points in latent space
latent_dim=256


# Generate random noise according to the number of samples specified
# The noise is a random vector from a normal distribution, with shape (n_samples, latent_dim)
noise = tf.random.normal([n_samples, latent_dim])

# Generate fake images from the generator using the random noise as input
x_fake = g_model(noise,training=False)

# Visualize the generated images
pyplot.figure(figsize=(25,25))

for i in range(n_samples):
    # define subplot
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    # plot single image
    pyplot.imshow(x_fake[i, :, :,0],cmap='gray')

pyplot.show()
pyplot.savefig("Fake_img_from_generator.png")
pyplot.close()

########################################################################
# Binary cross-entropy loss function for real/fake classification
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss function 
def discriminator_loss(real_output, fake_output):
    # Calculate loss on real images (compare real_output to a target of ones)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Calculate loss on fake images (compare fake_output to a target of zeros)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # Combine the two losses (real and fake)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss function
def generator_loss(fake_output):
    # Calculate loss on fake images
    # (the generator wants the discriminator to think they are real, hence target ones)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

########################################################################
# Define optimisers
# Adam optimizers with different learning rates for the generator and discriminator
generator_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

########################################################################
# Define training function
batch_size = 10 # 32 64 128 

# Define the training step
@tf.function
def train_step(images):
    # Generate random noise for the generator to create fake images
    noise = tf.random.normal([batch_size, latent_dim])

    # Record gradients for both the generator and discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images from the random noise
        generated_images = g_model(noise, training=True)

        # Get discriminator's classification of real and fake images
        real_output = d_model(images, training=True)
        fake_output = d_model(generated_images, training=True)

        # Calculate the loss for the generator and discriminator
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients for both models
    gradients_of_generator = gen_tape.gradient(g_loss, g_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, d_model.trainable_variables)

    # Apply the gradients to update the generator and discriminator optimizers
    generator_optimiser.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
    discriminator_optimiser.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    
    # Return the discriminator and generator losses for tracking
    return d_loss, g_loss

#########################################################################
#Define training loop
EPOCHS = 50 # TODO: Change the number of epochs (50 was too long)
batch_per_epoch=np.round(images.shape[0]/batch_size) # 9566 / 10

#number of sample images to display
n_samples=5

total_size=images.shape[0]


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(total_size).batch(batch_size)

def train(dataset, epochs):
    
    for epoch in range(epochs):
        count=0
        for image_batch in dataset:
            d_loss,g_loss=train_step(image_batch)
            if (count) % 25 == 0: # WHY 25? 
                print('>%d, %d/%d, d=%.8f, g=%.8f' % (epoch, count, batch_per_epoch, d_loss, g_loss))
            if (count) % 350 == 0:
                noise = tf.random.normal([n_samples, latent_dim])
                x_fake = g_model(noise,training=False)
                pyplot.figure(figsize=(25,25))
                for i in range(n_samples):
                    # define subplot
                    pyplot.subplot(5, 5, 1 + i)
                    pyplot.axis('off')
                    pyplot.imshow(x_fake[i, :, :,0],cmap='gray') # plot single image 
                # saves an image every 350 increment 
                pyplot.savefig('0511 Epoch{0} batch{1}.png'.format(epoch,count)) 
                pyplot.show()
                
                pyplot.close()

                # TODO: Saving model take too much space
                # just save one model version per epoch
                # USEFUL too see which 
                # filename = 'generator_model_%03d.h5' % (epoch) 
                # g_model.save(filename)

            count=count+1

train(train_dataset, EPOCHS)

#######look at generator images########
n_samples=5
noise = tf.random.normal([n_samples, latent_dim])
x_fake = g_model(noise,training=False)

pyplot.figure(figsize=(25,25))
for i in range(n_samples):
    # define subplot
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    # plot single image
    pyplot.imshow(x_fake[i, :, :,0],cmap='gray')
pyplot.show()
pyplot.close()

############SSIM#################
#since calculating SSIM for one image is computationally expensive, just choose the index of one image to calculate
#whichfake is the index of the sample image
whichfake=4

#create array to store SSIM values
ssim_noise=[]

#calculate SSIM for each training image
for i in range(images.shape[0]):
    ssim_noise.append( ssim(images[i,:,:,0], x_fake.numpy()[whichfake,:,:,0], 
                      data_range=np.max(x_fake.numpy()[whichfake,:,:,0]) - np.min(x_fake.numpy()[whichfake,:,:,0])))

#plot generated image and OASIS image that corresponds to the highest SSIM value
fig, axs = pyplot.subplots(2, 1, constrained_layout=True,figsize=(10,10))
axs[0].imshow(x_fake[whichfake, :, :, 0],cmap="gray")
axs[0].set_title('Generated image with max SSIM: {:.4f}'.format(np.max(ssim_noise)))

axs[1].imshow(images[ssim_noise.index(np.max(ssim_noise)), :, :, 0],cmap="gray")
axs[1].set_title('Closest OASIS image {:.0f}'.format(ssim_noise.index(np.max(ssim_noise))))

pyplot.show()