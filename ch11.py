import tensorflow as tf

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

# Batch normalization to adress exploding/vanishing gradient problems and more generally the problem
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