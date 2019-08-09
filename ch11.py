import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# choice of the activation function
# general ELU > leaky ReLU (+ its variants) > ReLU > tanh > logistic

# TensorFlow offers ELU implementation
hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.elu)

# TensorFlow does not have predefined leaky ReLU but easily defined:
def leaky_relu(z, name=None):
    return tf.maximum(0.01*z, z, name=name)
hidden1 = fully_connected(X, n_hidden1, activation_fn=leaky_relu)

# He initialization with ELU reduce vanishing/exploding gradients at the beginning of the training
# but no guarantee not come back later in the training

# BATCH NORMALIZATION: to adress exploding/vanishing gradient problems and more generally the problem
# that the distribution of each layer's input changes during the training as the parameters of 
# the previous layer change (internal covariate shift problem)
#  => add an operation just before the activation function simply zero-centering and normalizing the inputs,
# then scaling and shifting the result (using 2 new parameters per layer -> lets the model learn optimal scale and mean of the inputs)
# to 0-center and normalize, needs to estimate inputs' mean and sd -> evaluate mean and sd of current mini-batch (hence the name)
# at test time: no mini-batch -> use whole training set's mean and std
# -> reduce vanishing gradient problems
# -> less sensitive to weight initialization
# -> bigger learning rate can be used
# -> acts as regularizer
# TensorFlow provides batch_normalization() that simply centers and normalizes the inputs,
# but you must compute mean and std yourself (based on mini-batch data during the training, on full dataset during testing)
# and pass them as parameters; must also handle creation of scaling and offset parameters
# batch_norm() is more convenient: handles all of this; call it or directly tells fully_connected to use it
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
# -> will be either True or False; tells batch_norm() whether to use 
# current mini-batch mean and sd (during training) or the running averages it keeps track (during testing)

bn_params = {'is_training': is_training,
             'decay': 0.99,
             'updates_collections': None}
# batch_norm() uses exponential decay to compute the running averages -> needs a decay parameter (good value is close to 1)
# updates_collections = None -> update running averages right before it performs batch normalization during training (i.e. when is_training=True)
# if this parameter not set, TensorFlow will add the operations that update the running averages to a collection of operations that you must run yourself
             
hidden1 = fully_connected(X, n_hidden1, scope="hidden1", normalizer_fn=batch_norm, normalizer_params=bn_params)
hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", normalizer_fn=batch_norm, normalizer_params=bn_params)
logits= fully_connected(hidden2, n_outputs, activation_fn=None, scope="outputs", normalizer_fn=batch_norm, normalizer_params=bn_params)
             
# by default, batch_norm only centers, normalizes and shifts the inputs, does not scale them (gamma = 1)
# this makes sense for layers with no activation function or with ReLU activation, since next layer's weight
# can take care of scaling; but for any other activation function, add:
bn_params["scale"] = True # not needed if no activation function or ReLU

# to avoid repeating the same parameters, can create an argument scope using arg_scope()
# 1st param = list of functions; the other params will be passed automatically

with tf.contrib.framework.arg_scope([fully_connected], normalizer_fn=batch_norm, normalizer_params=bn_params):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
    
# execution phase almost the same, only need to change is_training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        #[...]
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op, feed_dict={is_training:True, X:X_batch, y:y_batch})  # set is_training to True
        accuracy_score = accuracy.eval(feed_dict={is_training:False, X:X_test_scaled, y:y_test}) # here is test run -> set is_training to False
        print(accuracy_score)
        
# GRADIENT CLIPPING: to lessen exploding gradient problem, clip gradients during backpropagation 
# so that they never exceed some threshold
# the minimize() function [used earlier] of the optimizer computes and applies the gradients; so instead use:
threshold = 1.0 # hyperparameter that can be tuned
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
# clip_by_value() -> operation to clip the gradients
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
# apply_gradients() -> apply the clipped gradients
training_op = optimizer.apply_gradients(capped_gvs)
# -> compute the gradients, clipped them between -1,1 and apply them

# TRANSFER LEARNING: reusing pretrained model
# [...] # <- construct the original model
with tf.Session() as sess:
    saver.restor(sess, "./my_original_model.ckpt")
    # [...] # <- train it on the new task
      
# but often want to reuse only a part of the trained model
# -> configure the Saver to restore only a subset of variables from the original model; e.g. to restor only hidden layers 1,2,3
# [...] # <- build new model with same definition as before for layers 1-3
init = tf.global_variables_initializer()
# get the list of all trainable variables just created with trainable=True (default), 
# keeps only those matching regular expression "hidden[123]"
reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[123]")
# create a dict to map the name of each variable in the original model to its name in the new model (generally the same)
reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
original_saver = tf.Saver(reuse_vars_dict) # saver to restore the original model
new_saver = tf.Saver() # saver to save the new model
with tf.Session() as sess:
    sess.run(init)
    original_saver.restore("./my_original_model.ckpt") # rest1ore hidden layers 1,2,3
    # [...] -> train the new model
    new_saver.save("./my_new_model.ckpt")# save the new model    
# NB: if model from another framework, weights and biases can be assigned manually (create nodes and assign the arbitrary values)
    
# FREEZING LOW-LEVEL LAYERS: makes high-level layers more easy to train
# give the optimizer the list of variables to train (exclude variables from lower layers):
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")    
training_op = optimizer.minimize(loss, var_list = train_vars) # -> layers 1 and 2 are now frozen

# the more data available, the more layers can be unfrozen

# frozen layers won't change -> cache the output of the topmost frozen layer for each training instance
# -> spped boost since training goes through the whole dataset many times
# (-> go through the frozen layers 1x/training instead of 1x/epoch)
# e.g., run the whole training set through the lower layers:
hidden2_outputs = sess.run(hidden2, feed_dict={X:X_train})
# during training build batches of hidden2_outputs instead of batches of training instances:
import numpy as np
import random as rnd
n_epochs = 100
n_batches = 500
for epoch in range(n_epochs):
    shuffled_idx = rnd.permutation(len(hidden2_outputs))
    hidden2_batches = np.array_split(hidden2_outputs[shuffled_idx], n_batches) 
    y_batches = np.array_split(y_train[shuffled_idx], n_batches)
    for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
        sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})
        
# if no existing model available and not a lot of labeled training data
# -> unsupervised training: train each layer 1 by 1 (starting from low-level) using unsupervised feature detection algorithm
# -> pretraining on auxiliary task (for which you can easily have enough labeled training data)        

# 5 ways to speed up the training
# 1) good initialization of the weights
# 2) good activation function
# 3) Batch Normalization
# 4) reuse of pretrained network
# 5) good optimizer (-> use Adam !)

# Momentum optimization (gradient as acceleration, not speed; take care of previous gradients) 
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# momentum value is an additional hyperparameter to tune, but 0.9 works well in practice

# Nesterov accelerated gradient: variant to momentum optimization (almost always faster)
# measure gardient of the cost function not at local position but slightly ahead in the direction of the momentum
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

# AdaGrad: scales down the gradient vector along the steepest dimension (accumulates the square of the gradients)
# decays the learning rate, but it does so faster for steep dimensions than for dimensions with gentler slopes
# -> adaptive learning rate; requires less tuning of the learning rate hyperparameter
# ! but should not use it when training NN because stops too early (works well for simple quadratic problems)

# RMSProp: fixes AdaGrad problems by accumulating only the gradients from most recent iterations
# use exponential decay in the first steps (decay rate as hyperparameter, usually set to 0.9)
# smoothing term epsilon to avoid division by 0
optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, momentum=0.9, decay=0.9, epsilon=1e-10)

# Adam = adaptive movement estimation => combines momentum optimization and RMSProp
# // momentum optimization: keeps track of exponentially decaying average of past gradients
# // RMSProp: keeps track of exponentially decaying average of past squared gradients
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # other default parameters usually work well
# also an adaptive learning rate algorithm -> requires less tuning, usually default value 0.001 can often be used


# training sparse models
# - clip small weights to 0
# - use strong L1 regularization
# - dual averaging (FTRL -> FTRLOptimizer)

# better than a constant learning rate: reduce learning rate during training
# (= learning schedules)
# - predetermined piecewise constant learning rate
# - performance scheduling
# - exponential scheduling
# - power scheduling (drops much more slowly than exponential)

# generally, favor exponential scheduling; ex. of TensorFlow implementation:
initial_learning_rate = 0.1
decay_steps= 10000
decay_rate = 1/10
# create a non trainable variable to keep track of the current training iteration number
global_step = tf.Variable(0, trainable=False)
# create the exponentially decaying learning rate
learning_rate = tf.train_exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
# pass the global_step variable -> will take care of incrementing it
training_op = optimizer.minimize(loss, global_step=global_step)

# NB: since AdaGrad, RMSProp and Adam automatically reduce learning rate during training, no need to add learning schedule !

# AVOID OVERFITTING: REGULARIZATION
# - early stopping
# - L1 and L2 regularization
# - dropout
# - max-norm regularization (constrains the weights of incoming connections)
# - data augmentation

# other efficient way to train very deep NN: add skip connections = add input of a layer to the output of a higher layer

# L1 and L2 regularization: add the appropriate regularization terms to the cost function, e.g. L1 regularization for 2 layers:
# [...]
base_loss = tf.reduce_mean(xentropy, name="xentropy")
reg_losses = tf.reduce_sum(tf.abs(weights1)) + tf.reduce_sum(tf.abs(weights2))
loss = tf.add(base_loss, scale*reg_losses, name="loss")

# not convenient when working with more layers !
# => many functions that create variables (e.g. get_variable() or fully_connected()) accept a *_regularizer argument
# for each created variable (e.g. weights_regularizer)
# => pass the function that takes weights as an argument and returns the corresponding regularization loss
# (e.g. the functions l1_regularizer(), l2_regularizer(), l1_l2_regularizer())
with tf.contrib.framework.arg_scope([fully_connected], weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs")

# then add the regularization losses to the overall loss
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([base_loss] + reg_losses, name="loss")

# DROPOUT = at every training step, every neuron as a probability p of being ignored during this training step
# dropout rate typically set to 50%
# but during testing, if p=50%, the neuron will be connected as twice as many neurons as during training
# -> multiply each connection weight by the keep probability 1-p
# (or divide each neuron's output by the keep probability during training)
# in TensorFlow: apply dropout function to the input layer and to the output of every hidden layer
# -> during the sampling, it randomly drops some items (setting them to 0) and divides the remaining items
# by the keep probability (does nothing at all after the training), e.g.
from tensorflow.contrib.layers import dropout
#[...] 
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
keep_prob = 0.5
X_drop = dropout(X, keep_prob, is_training=is_training)
hidden1 = fully_connected(X_drop, n_hidden1, scope="hidden1")
hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)
hidden2 = fully_connected(hidden1_drop, n_hidden2, scope="hidden2")
hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
logits = fully_connected(hidden2_drop, n_outputs, activation_fn=None, scope="outputs")
# !!! use dropout() from tf.contrib.layers and not the function from tf.nn
# because this latter does not turn off when not training !!!

# MAX_NORM REGULARIZATION
# constrains weights of incoming connections (clip them); 
# helps reduce overfitting and alleviate vanishing/exploding gradients
# not provided by TensorFlow, should be implemented:
threshold = 1.0
# clip weights variables along the 2nd axis so that each row vector has a maximum norm of 1
clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
clip_weights = tf.assign(weights, clipped_weights)
# then apply this operation at each training step
with tf.Session() as sess:
    # [...]
    for epoch in range(n_epochs):
        # [...]
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op, feed_dict={X:X_batches, y:y_batch})
            clip_weights.eval()
# then retrieve the weights
            
# cleaner solution is to write a function
# -> the function returns a parametrized max_norm() function to use like other regularizers
def max_norm_regularizer(threshold, axes=1, name="max_norm", collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        # does not require regularization loss term to add to overall loss
        # but need to be able to run the clip_weights operation at each training step -> add to a collection
        tf.add_to_collection(collection, clip_weights)
        return None # no regularization loss term
    return max_norm
max_norm_reg = max_norm_regularizer(threshold=1.0)
hidden1 = fully_connected(X, n_hidden1, scope="hidden1", weights_regularizer=max_norm_reg)

clip_all_weights = tf.get_collection("max_norm")
with tf.Session() as sess:
        # [...]
    for epoch in range(n_epochs):
        # [...]
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op, feed_dict={X:X_batches, y:y_batch})
            sess.run(clip_all_weights)
            
# recommandation default configuration:
# => He initialization
# => ELU activation function
# => Batch Normalization
# => Dropout
# => Adam optimization


