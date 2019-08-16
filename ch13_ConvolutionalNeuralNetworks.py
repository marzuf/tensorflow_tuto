# input image represented as 3D tensor of shape
# [height, width, channels]

# a mini-batch is represented as 4D tensor of shape
# [mini-batch size, height, width, channels]

# weights are represented as 4D tensors 
# bias terms are represented as 1D tensor
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_images

import matplotlib.pyplot as plt 

# CONVOLUTIONAL LAYER

# load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape # shape: (2, 427, 640, 3)

# create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 0] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line

# create graph with input X + conv. layer applying the 2 filters
X = tf.placeholder(tf.float32, shape = (None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides = [1,2,2,1], padding="SAME") #  stride of the sliding window for each dimension of input

# strides: 1st element can be used to specify a batch stride (skip some instances); 
# last element: channel stride (skip previous layer's feature map or channels)

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset}) # shape: (2, 214, 320, 2); stores a tensor 

plt.imshow(output[0, :, :, 0]) # plot 1st image's 1st feature map
plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.imshow(output[1, :, :, 0]) # plot 1st image's 1st feature map
plt.imshow(output[1, :, :, 1]) # plot 1st image's 2nd feature map
plt.show

# CNN requires huge amount of RAM especially during training because reverse pass of backpropagation
# requires all the intermediate values computed during forward pass
# (for prediction only need to store values of 2 consecutive layers)

# if memory issue: can try smaller mini-batch size, or reducing dimensionality (stride, remove layers), or use float16 instead of float32

# POOLING LAYER = subsample 
# -> reduce computational load, memory usage and # of parameters (reduce risk of overfitting)
# -> makes NN tolerate a little bit image shift (location invariance)
# has no weights, simply aggregate function to aggregate the inputs
# max-pooling is the most common 
# works on every channel independently (input depth = output depth)
# [but can alternatively be used to pool over depth dimensions, reducing # of channels but keeping spatial dimensions unchanged]

max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID") # use avg_pool() for average pooling

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})
plt.imshow(dataset[0].astype(np.uint8)) # original 1st image
plt.imshow(output[0].astype(np.uint8))  # output of the 1st image
plt.imshow(dataset[1].astype(np.uint8)) # original 2nd image
plt.imshow(output[1].astype(np.uint8))  # output of the 2ndimage

# other TensorFlow CNN operations:
# -> conv1d(): may be useful for language processing
# -> conv3d(): may be useful for 3D inputs 
# -> atrous_conv2d(): dilation (filter dilated by inserting rows/columns of 0)
# -> con2vd_transpose(): transpose CL = deconvolutional layer
# -> depthwise_conv2d(): applies every filter to every individual input channel independently (n filters on n' input channels -> produce n*n' feature maps) 
# -> separable_conv2d(): first acts like depthwise_conv2d then apply 1x1 convolution (-> possible to apply filters to arbitrary sets of input channels) 

# deconvolutional layer: performs upsampling by inserting 0 between the inputs (~conventional CL using fractional stride)
# useful e.g. in segmentation: in typical CNN, the feature maps get smaller and smaller, so if you want to output an image
# of the same size of the input, you have to upsample