# MLP implemented with mini-batch gradient descent
# 1) construction phase -> build the TensorFlow graph
# 2) execution phase -> actually run the graph to train the model

# CONSTRUCTION PHASE
# -> create placeholder (for input and target)
# -> create function to build a neuron layer
# -> create the NN model
# -> define the cost function
# -> create the optimizer
# -> create performance measure
# -> create initializer
# -> create Saver
import tensorflow as tf
n_inputs = 28*28 # MNIST data set
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
# 0) create placeholder
# set size to None since we don't know yet batch size
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")  # input layer, will be replaced with training batch
y = tf.placeholder(tf.int64, shape=(None), name='y')
# create the actual NN
# all the instances in a training batch will be processed simultaneously
# create 2 hidden layers with ReLU and output layer with softmax
# -> to avoid repeat code, create a function that creates the layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name): # create name scope (-> will look nicer in TensorBoard)
        n_inputs = int(X.get_shape()[1]) # retrieve nbr of inputs
        stddev = 2/np.sqrt(n_inputs) # help faster convergence
        init = tf.truncate_normal((n_inputs, n_neurons), stddev=stddev) # contain weights between each input and each neuron, hence n_inputs*n_neurons
        # -> use of truncated normal ensures no large weights which could slow down the training
        W = tf.Variables(init, name="weights") # random init. is important to break symmetry !!!
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z
# a) create the NN
with tf.name_scope("dnn"): # use a name scope for clarity
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")
    # logits hold the output before the softmax; for optimization reasons, handle the softmax computation later
# NB: TensorFlow normally provides such function and normally no need to create own neuron_layer(), for example:
from tensorflow.contrib.layers import fully_connected
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1") # default activation is ReLU
    hidden1 = fully_connected(hidden1, n_hidden2, scope="hidden2") 
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

# b) define the loss function    
# use cross-entropy as loss functionfor the training; 
# penalize models that estimate low probability for the target class
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # computes cross-entropy based on logits
    # expect labels in the form of integers ranging from 0 to #class-1 (in our case: 0-9)
    # equivalent to applying softmax activation and then computing cross-entropy but is more efficient
    # and takes care of corner cases like logits=0 -> do not apply softmax earlier
    # similar function softmax_cross_entropy_with_logits that takes labels in form of one-hot vectors
    loss = tf.reduce_mean(xentropy, name="loss") # compute the mean cross-entropy over all instances

# c) define the optimizer
# we have the NN model and the cost -> now define the optimizer that will tweak the parameters to minimize the cost function
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
# d) specify how to evaluate the model (performance measure)
# here accuracy is used as performance measure
# for each instance, determine if the prediction is correct (does highest logit correspond to target class ?)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1) # returns 1D tensor full of boolean values (cast to float to take the mean)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # -> give network overall accuracy
    
# e) nodes for initialization and saving
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# EXECUTION PHASE
# ! from Stanford course: when normalizing the data, e.g., the mean is computed on training data only, but substracted to all data (i.e. incl. validation and test) 
# TensorFlow helper that fetches the data, scales it (between 0 and 1), shuffles it, and provides a simple function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")

# A) define the number of epochs and size of mini-batch
n_epochs = 400
batch_size = 50 # -> number of batches will be given by # of examples/batch_size
# B) train the model
# -> open the session
# -> initialize the variables
# -> at each epoch
# -> ... iterate over the batches
# -> ...... run the training by feeding the graph with current batch data (inputs and targets)
#-> ... at the end of the epoch: evaluates the model with last mini-batch and full training dataset
with tf.Session() as sess:
    init.run() # runs the init node to initialize all the variables
    for epoch in range(n_epoch):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch}) # train with current mini-batch
        # end of the epoch: evaluate the model with last mini-batch and full training dataset # ?? full test dataset ??
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test =  accuracy.eval(feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print(epoch, " - Train accuracy:", acc_train, " - Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt") # save parameters to disk
    
# now that the model is trained, can be used to make predictions
# !!! use the same feature scaling as for the training data !!!
with tf.Session() as sess:
    saver.restor(sess, "./my_model_final.ckpt")
    X_new_scaled = [...] # some new images scaled from 0 to 1
    Z = logits.eval(feed_dict={X:X_new_scaled})
    # pick the class with highest value (np.argmax()) to have the predicted class
    y_pred = np.argmax(Z, axis=1)
    # -> if you want probabilities, apply softmax() to the logits
    
    