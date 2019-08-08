import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

f = x*x*y + y + 2

# => this already creates a simple graph
# this does not actually perform any computation
# but just creates a computation graph
# even the variables are not initialized
# To evaluate the graph, need to open a TensorFlow session and use it to initialize the
# variables and evaluate f

# create the TensorFlow session
sess = tf.Session()
# initialize the variables
sess.run(x.initializer)
sess.run(y.initializer)
# evaluate the variables
result = sess.run(f)
print(result)
# closes the session
sess.close()


# to avoid repeatedly writing sess.run() all the time, can be written as
with tf.Session() as sess:
    x.initializer.run() # equivalent to: tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval() # equivalent to tf.get_default_session().run(f)
# the session is automatically closed at the end of the block
    
# instead of manually initializing every single variable individually:
init = tf.global_variables_initializer()
with tf.Session() as sess:
        init.run() # actually initializes all the variables
        result = f.eval()
        
        
# A TensorFlow program is typically split in 2 parts:
        # 1) build the computation graph (construction phase)
        # => builds a CG representing the ML model and the computations required to train it
        # 2) run the graph (execution phase)
        # => typically runs a loop that evaluates a training step repeatedly, gradually improving model parameters
        
        
# when you evaluate a node, TF automatically determines the set of nodes
# that it depends on and evaluates these nodes automatically
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())
# ! will not reuse results of the previous evaluation(s) (here w and x will be evaluate twice)
   
# all node values are dropped between graph runs except variable values
   
# to evaluate y and z efficiently in one single run:
with tf.Session() as sess:
    y_val, z_val = sess.run([y,z])
    print(y_val)
    print(z_val)
    
# TensorFlow operations (ops) can take any number of inputs and produce any number of outputs
# constants and variables take no input (source ops)
    
# input and outputs are multidimensional arrays (tensors)
    
# example of linear regression (using Normal equation)
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data] # Translates slice objects to concatenation along the second axis.

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y") # reshape from 1D array to column vector (-1 means unspecified)
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    
# advantage over numpy implementation: TensorFlow will automatically run it on GPU card if one available


# linear regression with self-implemented gradient descent
# when using gradient descent, do not forget to normalize the input !

scaled_housing_data_plus_bias = housing_data_plus_bias  # assume normalization is done !

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name="theta") # create a node that will generate random values
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta-learning_rate * gradients) # create a node that will assign a new value to a variable

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)    
    for epoch in range(n_epochs):  # the main loop executes the training step
        if epoch % 100 == 0:
            print("Epoch ", epoch, " - MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    
    
# TensorFlow autodiff feature: automatically and efficiently compute the gradients
gradients = tf.gradients(mse, [theta])[0]
# => takes an ops (here: mse) and a list of variables (here: theta) and creates a list of ops
# (one ops per variable) to compute the gradients of the op wiht regards to each variable

# TensorFlow performs reverse-mode autodiff

# TensorFlow makes things even easier: also provides a number of optimizers:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


# Implement mini-batch gradient descent
# -> replace X and y at every iteration with the next mini-batch
# => use placeholder nodes -> they don't perform any computation, 
# they just output the data you tell to output at running time 
# (typically used to pass training data to TensorFlow during training)

A = tf.placeholder(tf.float32, shape=(None,3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A:[[4,5,6], [7,8,9]]})
print(B_val_1)    
print(B_val_2)

# implement mini-batch gradient descent:
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None,1), name="y")

# define the batch size and compute total number of batches
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

# in the execution phase, fetch the mini-batches one by one

def fetch_batch(epoch, batch_index, batch_size):
    #... # load data from disk etc.
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            best_theta = theta.eval() # no need to pass X nor y when evaluating theta since does not depend on either of them
            
            
# save and restore models
# save parameters to be able to use them later
# save checkpoints so that if it crashes you can restart from last checkpoint
# create a Saver node and then call save()

# [...]
init = tf.global_variables_initializer()
saver = tf.train.Saver() # create a Saver node

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0: # save checkpoint every 100 epochs
            save_path = saver.save(sess, "my_model.ckpt")
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, "my_model_final.ckpt")
    
# to restore the model
with tf.Session() as sess:
    saver.restore(sess, "my_model_final.ckpt")
    
# by default, saves and restores all variables under their own name
# but can specify which variables to save and the names to use
saver = tf.train.Saver({"weights", theta}) # save and restore only theta under the name 'weights'

# visualization with TensorBoard
# -> write grpah definition and some training stats in a log directory
# use a different log directory everytime the program is run (i.e. include a timestamp in the directory name), ex:
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

# then add at the end of construction phase
mse_summary = tf.summary.scalar('MSE', mse)
# -> creates a node in the graph that will evaluate the MSE value and write it to a TensorBoard-compatible binary log string called a 'summary'
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# -> creates a FileWriter that will be used to write summaries to logfiles in the log directory
# the 2nd (optional) parameter is the graph you want to visualize
# upon creation, FileWriter creates the log directory (and parent directories if needed), and writes graph definition in a binary logfile called 'events file'

# change the execution phase to evaluate the mse_summary node regularly during training (e.g. every 10 mini-batches)
# [...]
for batch_index in range(n_batches):
    X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
    if batch_index % 10 == 0:
        summary_str = mse_summary.eval(feed_dict={X:X_batch, y:y_batch})
        step = epoch * n_batches + batch_index
        file_writer.add_summary(summary_str, step)
    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
# [...]
    
# add the end of the program, close the FileWriter
FileWriter.close()    

# then to visualize with TensorBoard, enter the following command in a terminal
# tensorboard --logdir tf_logs/
# this indicates which port is being listened -> can be opened in a browser

# to help visualization, you can create name scopes to group related nodes

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
print(error.op.name)
# loss/sub
print(mse.op.name)
# loss/mse
# -> name of each op defined within the scope is prefixed with "loss/"
# (-> in TensorBoard graph, will appear inside the "loss" namespace, that appears collapsed by default)


# Modularity
# e.g. you want to create a graph that adds the output of 2 ReLU
# this can be written:
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0., name="relu1")
relu2 = tf.maximum(z2, 0., name="relu2")

output = tf.add(relu1, relu2, name="output")

# -> instead, create a function that build a ReLU

def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  # if the name already exists, TF will add "_[index]" to make the name unique
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)] 
output = tf.add_n(relus, name="output")

# Sharing variables
# to share variable between various components of the graph, create it first and then pass it as a paramter to the functions that need it
# example to control ReLU threshold:
def relu(X, threshold):
    with tf.name_scope("relu"):
        #[...]
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)] 
output = tf.add_n(relus, name="output")


# if there are many parameters, people usually create a Python dictionary containing 
# all variables in their model and pass it to every function
# or create a class for every module (e.g. a ReLU class), using class variables to handle the shared parameters

# or set the shared variables as an attribute of the relu() function upon the first call:
def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        # [...]
        return tf.maximum(z, relu.threshold, name="max")

# but cleaner and more modular way offered by TensorFlow:
# use get_variable() -> create the shared variable if does not exist or reuse it if already exists
# creating or reusing is controlled by an attribute of the current variable_scope()

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
    # -> create a variable named relu/threshold (a scalar since shape=() and use 0.0 as initial value)
# if the variable has already been created by an earlier call to get_variable -> raise an exception
# (prevents reusing variables by mistake)
# -> to reuse a variable, need to be explicitly said by setting reuse attribute to True:
with tf.variable_scope("relu", reuse=True):
    threshold = tf.get_variable("threshold")
    # -> fetch the existing threshold variable, raise an exception if not already exist or not created with get_variable earlier
# you can set reuse attribute inside the block using reuse_variables()
with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")
    
# -> once reuse set to True, cannot be set to False within the block
# -> if other variable scopes inside this scope, will inherit reuse=True also
# -> only variables created with get_variable can be used this way
    
    
# => make the relu function having access to threshold parameter without passing it as parameter:
def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold") # reuse existing variable
        #[...]
        return tf.maximum(z, threshold, name="max")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"): # create the variable
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
relus = [relu(X) for i in range(5)] 
output = tf.add_n(relus, name="output")

# variables created using get_variable() are always named using name of variable_scope as a prefix
# (e.g. relu/threshold)
# but for all other nodes (incl. tf.Variable()) the variable scope acts like a new name scope (a suffix is added if needed)
# -> all nodes created in the code here above - except the threshold variable - will have name prefixed with relu_1/ ... to relu_5/

# to have the threshold variable defined inside relu() (to have all the ReLU code at the same place),
# can create the variable upon the first call and then reuse it

def relu(X):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
    #[...]
    return tf.maximum(z, threshold, name="max")
# inside the function, no need to know if reuse is True or False (does not have to care about name scopes)    
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index>=1)) as scope: # create the variable
        relus.append(relu(X))
output = tf.add_n(relus, name="output")


