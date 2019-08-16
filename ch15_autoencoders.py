### AUTOENCODER AS PCA
# if the autoencoder (AE) uses only linear activations and the cost function is MSE
# then it ends up performing PCA

import tensorflow as tf
import numpy as np
g = tf.Graph()
with g.as_default():
    t1 = tf.constant([1,2,3,4])
    X = tf.placeholder(tf.int32, shape=None)
    Y = tf.placeholder(tf.int32, shape=None)
    loss = tf.add(X, Y)
    init = tf.global_variables_initializer()
with tf.Session(graph=g) as sess:
    z1 = t1.eval()
    z2 = sess.run(t1)
    print(z1)  #[1,2,3,4]
    print(z2)  #[1,2,3,4]
    # t1.run() # tensor object has no attriubte 'run'
    z3 = sess.run(loss, feed_dict={X:1, Y:2})
    print(z3) # 3
    z4 = loss.eval(feed_dict={X:1, Y:2})
    print(z4) # 3
    #z5 = loss.run(feed_dict={X:1, Y:2}) # Tensor object has no attribute 'run'
    #print(z5) # 3
    init.run() # no error; init is a tf.Operation

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_inputs = 3 # 3d inputs
n_hidden = 2 # 2d codings

n_outputs = n_inputs # # outputs = # inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=None)  # linear activation -> PCA
outputs = fully_connected(hidden, n_outputs, activation_fn=None) # linear activation -> PCA

reconstruction_loss = tf.reduce_mean(tf.square(outputs-X))  # MSE -> PCA

optimizer = tf.train.AdampOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

X_train, X_test  = [...] # load dataset

n_iterations = 1000

codings = hidden # the output of the hidden layer provides the codings

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train}) # no labels, feed only with the X
    codings_val = codings.eval(feed_dict={X: X_test})
    
    
### STACKED AUTOENCODER
# typically symmetrical wrt to the central layer (coding layer)    
n_inputs = 28*28  # for MNIST DATA
n_hidden1  = 300
n_hidden2 = 150
n_hidden3 = n_hidden1  # symmetric
n_outputs = n_inputs 
# implementation similar to MLP
# here: He initialization, ELU activation, l2 regularization
learning_rate = 0.01
l2_reg = 0.001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# to call all fully_connected() with the same argument list
with tf.contrib.framework.arg_scope(
                    [fully_connected],
                    activation = tf.nn.elu,
                    weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                    weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)):
    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2)                        
    hidden3 = fully_connected(hidden2, n_hidden3)
    outputs = fully_connected(hidden3, n_outputs, activation_fn=None)
    
reconstruction_loss = tf.reduce_mean(tf.square(outputs-X)) # MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  # retrieve the loss from regularization
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.optimize(loss)

init = tf.global_variables_initializer()

# train the model normally (without using labels)
n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size) # NB: y_batch (the labels) not used
            sess.run(training_op, feed_dict={X: X_batch})

### STACKED AUTOENCODER - TYING WEIGHTS: when symmetrical, tie decoder layer weights to encoder layer weights
# cumbersome to implement with the fully_connected(), easier to define the layers manually
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name="weights3") # tied weights
weights4 = tf.transpose(weights1, name="weights4") # tied weights

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# faster to train one shallow AE at a time, then stack all of theme into a single stacked AE
# similar to previous ones, could look like

# build the whole stacked AE normally
# here, the weights are not tied
optimizer = tf.train.AdamOptimizer(learning_rate)

# 1st phase: create an output layer that skips layers 2 and 3 (learn to reconstruct the inputs)
# train weights and biases for hidden layer 1 and output layer
# makes outputs of layer 1 ~ inputs
with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)

# 2nd phase: train weights and biases for hidden layers 2 and 3 (learn to reconstruct the ouptut of hidden layer 1)
# makes outputs of layer 3 ~ output of layer 1
with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
    train_vars = [weights2, biases2, weights3, biases3]
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars)


### VISUALIZING RECONSTRUCTIONS
n_test_digits = 2
X_test = mnist.test.images[:n_test_digits]

with tf.Session() as sess:
    [...] # train the autoencoder
    outputs_val = outputs.eval(feed_dict={X:X_test})
    
    def plot_image(image, shape=[28,28]):
        plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
        plt.axis("off")
        
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index*2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index*2 + 2)
        plot_image(outputs_val[digit_index])
        
### VISUALIZING FEATURES
# create images where pixel intensity = weight of the connection
with tf.Session() as sess:
    [...] # train the AE
    weights1_val = weights1.eval()
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plot_image(weights1_val.T[i])
        
### DENOISING AUTOENCODERS
# force AE to learn useful features by adding noise to the input
# add noise to the inputs, reconstruction loss calculated based on the original inputs
# v1: add Gaussian noise
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_noisy = X + tf.random_normal(tf.shape(X))
[...]
hidden1 = activation(tf.matmul(X_noisy, weights1) + biases1)
[...]
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
# v2: "dropout" (inputs randomly switched off)
from tensorflow.contrib.layers import dropout
keep_prob = 0.7
is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
X = tf.placeholder(tf.float32, shape=[None, n_inputs]) 
X_drop = dropout(X, keep_prob, is_training=is_training)
[...]
hidden1 = activation(tf.matmul(X_drop, weights1) + biases1)
[...]
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
[...]
# -> during training, do not forget to set is_training to True
sess.run(training_op, feed_dict={X:X_batch, is_training: True})
# for testing, need to be False, but no need to set explicitly as set by default

### SPARSE AUTOENCODERS
# add terms to cost function; e.g. limit the number of significant active neurons
# 1) compute actual sparsity = average activation of each neuron in the coding layer over whole training batch
# 2) penalize neurons that are too active: add sparsity loss to cost function
# for sparsity loss, better to use KL divergence, stronger gradients than MSE

def kl_divergence(p, q):
    return p * tf.log(p/q) + (1-p) * tf.log((1-p)/(1-q))

learning_rate = 0.01
sparsity_target = 0.1
# can add sparsity weight hyperparameter (relative importance of sparsity loss wrt to reconstruction loss)
sparsity_weight = 0.2 
[...]  # build a normal AE
optimizer = tf.train.AdamOptimizer(learning_rate)

hidden1_mean = tf.reduce_mean(hidden1, axis=0)
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target,  hidden1_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE
loss = reconstruction_loss + sparsity_loss * sparsity_weight
training_op = optimizer.minimize(loss)

# ! activations of the coding layer must be > 0 and < 1
# can use logistic activation function for the coding layer:
hidden1 = tf.nn.sigmoid(tf.matmul(X, weights1) + biases1)
# to speed up convergence: use cross-entropy instead of MSE
# but to use it, normalize inputs to make them ranging from 0 to 1
logits = tf.matmul(hidden1, weights2) + biases2
outputs = tf.nn.sigmoid(logits)
reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))
# outputs not needed for the training (only for to look at the reconstructions)

### VARIATIONAL AUTOENCODERS
# - probabilistic (outputs partly determined by chance)
# -generative (new instances that look like sampling from training set)
# instead of directly producing a coding -> produces a mean coding and stdv -> actual coding sampled from  Gaussian
# push the codings to gradually migrate within the coding space (latent space) to occupy a spherical regions similar to cloud of Gaussian points
# cost function with 2 parts
# 1) usual reconstruction loss (output~input)
# 2) latent loss (KL divergence) (codings~Gaussian)

# equations for latent loss:
eps = 1e-10 # smooting term to avoid computing log(0)
latent_loss = 0.5 * tf.reduce_sum(tf.square(hidden3_sigma) + tf.square(hidden3_mean) - 1 - tf.log(eps + tf.square(hidden3_sigma)))

# one common variant is to train the encoder to output gamma=log(sigma^2) rather than sigma;
# then juste compute sigma = exp(gamma/2)
# makes simga easier to capture sigmas of different scales thus helps speed up convergence
# latent loss ends up a bit simpler
latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)

# here below, example of a variational autoencoder using the log(sima^2) variant

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_inputs = 28*28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.001

with tf.contrib.framework.arg_scope(
                    [fully_connected],
                    activation_fn=tf.nn.elu,
                    weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
    X = tf.placeholder(tf.float32, [None, n_inputs])
    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2)
    hidden3_mean = fully_connected(hidden2, n_hidden3, activation_fn=None)
    hidden3_gamma = fully_connected(hidden2, n_hidden3, activation_fn=None)
    hidden3_sigma = tf.exp(0.5 * hidden3_gamma)
    noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
    hidden3 = hidden3_mean + hidden3_sigma * noise
    hidden4 = fully_connected(hidden3, n_hidden4)
    hidden5 = fully_connected(hidden4, n_hidden5)
    logits = fully_connected(hidden5, n_outputs, activation_fn=None)
    outputs = tf.sigmoid(logits)
    
reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))
latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
cost = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(cost)
init = tf.global_variables_initializer()


# now use the variational AE to generate images that look like handwritten
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")

import matplotlib.pyplot as plt 

n_digits = 60
n_epochs = 50
batch_size = 150

with tf.Session() as sess:
    init.run()
    #for epoch in range(n_epochs):
    for epoch in range(5):
        n_batches = mnist.train.num_examples // batch_size
        # for iteration in range(n_batches):
        for iteration in range(5):            
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch})
    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})
    
# look at the digits produced by the AE
for iteration in range(n_digits):
    plt.subplot(n_digits, 10, iteration+1)
    plot_image(outputs_val[iteration])
    