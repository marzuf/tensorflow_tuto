import tensorflow as tf
import numpy as np

g = tf.Graph()

## define the computation graph
with g.as_default():
    # define tensors
    t1 = tf.constant(np.pi)
    t2 = tf.constant(list(range(1,5)))
    t3 = tf.constant([[1,2],[2,3]])
    
    # get their ranks
    r1 = tf.rank(t1) # rank() returns a tensor -> should evaluate the tensor to get the value
    r2 = tf.rank(t2) 
    r3 = tf.rank(t3) 
    
    
    # get their shape
    s1 = t1.get_shape() # returns a TensorShape object
    s2 = t2.get_shape() 
    s3 = t3.get_shape() 
    print("Shape of t3: " + str(s3))
    
# "Session" => environment in which operations and tensors can be evaluated
# Session() can receive an existing graph in argument, otherwise the default graph
with tf.Session(graph=g) as sess:                   # need to specify the graph to retrieve the tensors !   
    print("Rank of t3: " + str(r3.eval()))          # eval() must be run inside a session !
    
# For building and compiling a TensorFlow graph:
    # 1) Initiate a new, empty computation graph
    # 2) Add nodes (tensors and operations) to the graph
    # 3) Execute the graph:
    #   - start a new session
    #   - initialize the variables
    #   - run the computation graph
    
# tf.Session.run() -> to run both tensors and operations
    
# placeholders -> do not contain any data; shape and stype should be decided at definition
# shape: for the varying dimension, can be set to None
with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[], name="tf_a")
    tf_b = tf.placeholder(tf.int32, shape=[], name="tf_b")
    
    r1 = tf_a - tf_b
    
with tf.Session(graph=g) as sess:
    feed = {tf_a:1, tf_b:2}    
    print("r1 = " + str(sess.run(r1, feed_dict=feed)))
    
# VARIABLES in TensorFlow = special type of tensor objects
# need to be intialized with a tensor of values !
# e.g. allow to store and update parameters during the training
# 2 ways to define variables:
    # tf.Variable(...)
    # -> shape and type set according to the input
    # tf.get_variable(name, ...)
    # -> to reuse or create variable
    # -> shape and type should be set explicitly
    # -> can add regularizer

# the initial values are not set until the graph is launched and explicitly run the initializer
# (required memory for a graph not allocated until variable intialization)

import tensorflow as tf
import numpy as np

g1 = tf.Graph()
with g1.as_default():
    w = tf.Variable(np.array([[1,2,3,4], [5,6,7,8]]), name="w")  # shape and type are inferred
    print(w)
    
# variable initialization: allocate memory and assign initial values
# global_variables_initializer() -> returns an operator for initializing all the variables existing in a graph
with tf.Session(graph=g1) as sess:
    # print(sess.run(w)) # -> will raise an error since w is not initialized !
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))

# the initializer operator can also be stored within an object
# ! but be careful to place it AFTER the variables he should initialized in graph definition
g2 = tf.Graph()    
with g2.as_default():
    w1 = tf.Variable(np.array([[1,2,3,4], [5,6,7,8]]), name="w1")  # shape and type are inferred
    init_op = tf.global_variables_initializer()
    w2 = tf.Variable(np.array([[1,4], [5,6]]), name="w2")  # shape and type are inferred
    print(w)
with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print(sess.run(w1))
    # print(sess.run(w2)) -> error, not initialized
    
# VARIABLE SCOPES
import tensorflow as tf
g = tf.Graph()
with g.as_default():
    with tf.variable_scope("net_A"):
        with tf.variable_scope("layer-1"):
            w1 = tf.Variable(tf.random_normal(shape=(10,4), name="weights"))
    print(w1)
# <tf.Variable 'net_A/layer-1/Variable:0' shape=(10, 4) dtype=float32_ref>
# -> variable names are prefixed with their nested scopes
    
# REUSING VARIABLES: use data from one source as input tensor and from another source later
# => example with a classifier
    
import tensorflow as tf
###############################
# Helper functions
###############################
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights', shape=(data_shape[1], n_classes), dtype=tf.float32)
    bias = tf.get_variable(name='bias', initializer=tf.zeros(shape= n_classes))
    logits = tf.add(tf.matmul(data, weights), bias, name="logits")
    return logits, tf.nn.softmax(logits)

def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1], n_hidden)), name="w1")    
    b1 = tf.Variable(tf.zeros(shape=(n_hidden)), name="b1")    
    hidden = tf.add(tf.matmul(data, w1), b1, name="hidden_pre-activation")
    w2 = tf.Variable(tf.random_normal(shape=(n_hidden, data_shape[1])), name="w2")    
    b2 = tf.Variable(tf.zeros(shape=(n_hidden)), name="b2")    
    output= tf.add(tf.matmul(hidden, w2), b2, name="output") # "+" is overloaded but with add() we can specify the name, can be useful
    return output, tf.nn.sigmoid(output)
###############################
# Build the graph
###############################
batch_size = 64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name="tf_X")
    # build the generator
    with tf.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)
    # build the classifier
    with tf.variable_scope("classifier") as scope:
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))  # -> to build the network
        # reuse the classifier for generated data
        scope.reuse_variables() # -> avoid creating new variables !
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size)) # -> the graph variables here are used, not created

# reusing variables can also be done with reuse=True parameter
with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name="tf_X")
    # build the generator
    with tf.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)
    # build the classifier
    with tf.variable_scope("classifier"):
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))  # -> to build the network
    # reuse the classifier for generated data    
    with tf.variable_scope("classifier", reuse=True):        
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size)) # -> the graph variables here are used, not created


# BUILDING A SIMPLE REGRESSION MODEL

import tensorflow as tf
import numpy as np
# 1) create the graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)
    # placeholders
    tf_x = tf.placeholder(shape=(None), dtype=tf.float32, name="tf_x")    
    tf_y = tf.placeholder(shape=(None), dtype=tf.float32, name="tf_y")
    # define the variables (model parameters)
    weight = tf.Variable(tf.random_normal(shape=(1,1), stddev=0.25), name="weight")
    bias = tf.Variable(0.0, name="bias")
    # build the model
    y_hat = tf.add(weight*tf_x, bias, name="y_hat")
    # compute the cost
    cost = tf.reduce_mean(tf.square(tf_y-y_hat), name="cost")
    # train the model
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name="train_op")
    
# 2) open a session and execute the graph and train the model
# create random toy dataset for regression
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
#    for t in x:
#        r = np.random.normal(loc=0.0, scale=(0.5+t*t/3), size=None)
#        y.append(r)
    y = [np.random.normal(loc=0.0, scale=(0.5+t*t/3), size=None) for t in x]        
    return x, 1.726*x-0.84 + np.array(y)

x, y = make_random_data()
plt.plot(x,y,'o'); plt.show()

# split train/test
x_train, x_test, y_train, y_test = x[:100,], x[100:], y[:100], y[100:]

n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    # train the model for n_epochs
    for epoch in range(n_epochs):
        # will do 2 tasks: execute an operator (execute train operator) and evaluate a tensor (calculate training cost)
        c, _ = sess.run([cost, train_op], feed_dict={tf_x:x_train, tf_y:y_train})
        training_costs.append(c)
        if not epoch % 50:
            print("Epoch %4d : %4f" % (epoch, c))
        
plt.plot(training_costs); plt.show()

# the same can be done by using the name of the variables
# the tensors have suffix ':0', but not the operators !
# Tensor names must be of the form "<op_name>:<output_index>".
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    # train the model for n_epochs
    for epoch in range(n_epochs):
        # will do 2 tasks: execute an operator (execute train operator) and evaluate a tensor (calculate training cost)
        c, _ = sess.run(['cost:0', 'train_op'], feed_dict={'tf_x:0':x_train, 'tf_y:0':y_train})
        training_costs.append(c)
        if not epoch % 50:
            print("Epoch %4d : %4f" % (epoch, c))
            
            
# saving and restoring a model
            
with g.as_default():
    saver = tf.train.Saver()  # <- add here to save !
    
n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    # train the model for n_epochs
    for epoch in range(n_epochs):
        # will do 2 tasks: execute an operator (execute train operator) and evaluate a tensor (calculate training cost)
        c, _ = sess.run([cost, train_op], feed_dict={tf_x:x_train, tf_y:y_train})
        training_costs.append(c)
        if not epoch % 50:
            print("Epoch %4d : %4f" % (epoch, c))
        
    saver.save(sess, './_ch14_trained-model')
     
 # 3 files are created with extensions: .meta, .index, .data
# to restore a built model:
# 1) rebuild the graph; 2) restor the saved variables
# tf.train_import_meta_graph() recreates the graph
# after re-creating the graph, use new_saver object to restore the parameters

g2 = tf.Graph()

with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph("./_ch14_trained-model.meta")
    new_saver.restore(sess, "./_ch14_trained-model")
    # MUST use the name here !
    y_pred = sess.run('y_hat:0', feed_dict={'tf_x:0':x_test})


x_arr = np.arange(-2, 4, 0.1)
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph("./_ch14_trained-model.meta")
    new_saver.restore(sess, "./_ch14_trained-model")
    # MUST use the name here !
    y_arr = sess.run('y_hat:0', feed_dict={'tf_x:0':x_arr})


plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr.T[:,0], '-r', lw=3)
plt.show()
    
# transforming tensors in multidimensional data arrays
# tf.get_shape()
# tf.reshape()
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    arr = np.array([[1.,2.,3.,3.5],
                    [4.,5.,6.,6.5],
                    [7.,8.,9.,9.5]])
    T1 = tf.constant(arr, name="T1")
    print(T1)
    s = T1.get_shape() # cannot be sliced or indexed -> should be converted to list using as_list()
    T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0],)))
    print(T3)


# 3 ways to transpose numpy array
# arr.T
# arr.transpose()
# np.transpose(arr)

# to tranpose array in TensorFlow:
# tf.transpose()
# 'perm' argument to change order of dimensions
with g.as_default():
    T6 = tf.transpose(T1, perm=[0,1],  name="T6")

# split() -> to split a tensor into a list of subtensors; output is a list of tensors
with g.as_default():
    T6_split = tf.split(T6, num_or_size_splits=3, axis=0, name="T6_split")
    
# concat() -> concatenation 
g = tf.Graph()
with g.as_default():
    t1= tf.ones(shape=(5,1), dtype=tf.float32, name='t1')
    t2= tf.zeros(shape=(5,1), dtype=tf.float32, name='t2')
    t3 = tf.concat([t1,t2], axis=0, name="t3")

# control flow mechanics in building graph
# !!! i use "normal" if-else in the graph
# only one branch will have been called
# the comptutation graph is static -> remains unchanged during execution process
# => needs to be wrapped in TensorFlow specific functions

# tf.case()
# tf.cond()
# tf.while_loope()

with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, shape=None, name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name='tf_y')
    res = tf.cond(tf_x < tf_y,  
                      lambda: tf.add(tf_x,tf_y, name='result_add'), 
                      lambda: tf.subtract(tf_x,tf_y, name='result_sub'))

with tf.Session(graph=g):
    print(res.eval(feed_dict={tf_x:1., tf_y:2.}))
    print(res.eval(feed_dict={tf_x:2., tf_y:1.}))
    
