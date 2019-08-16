import tensorflow as tf
import numpy as np
# very simple RNN without using TensorFlow implementation
# implements layer of 5 recurrent neurons with tanh activation function
# unrolled for 2 time steps

# (memory) cell = part of a NN that preserves some state across time steps
# (a recurrent neuron or a layer of recurrent neurons)

n_inputs = 3
n_neurons = 5

# create placeholder (1 for each time step)
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
# same weights and bias terms shared by both layers
# feed inputs at each layer and get outputs from each layer
init = tf.global_variables_initializer()

# to run the model, need to feed inputs at both time steps !
# mini-batch:
#        instance 0, instance 1, instance 2, instance 3
X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]]) # t=0
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]]) # t=1
# mini-batch with 4 instances, each with an input sequence composed of 2 inputs
# Y0_val and Y1_val contain the outputs of the network at both time steps
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0:X0_batch, X1:X1_batch})
    
print(Y0_val) # output at t=0
print(Y1_val) # output at t=1

# using TensorFlow static_rnn() which creates an unrolled RNN by chaining cells:

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# -> "cell factory": like creating copies of the cell to build the unrolled RNN (1 for each time step)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0,X1], dtype=tf.float32)
# -> input: the cell factory and the input tensors
# calls the cell factory's __call__() function once per input -> creates 2 copies of the cell with shared weights and biases
# (each copy contains a layer of 5 recurrent neurons) 
# returns:
# 1) list containing the output tensor for each time step
# 2) the final states of the network
# (when using basic cells, final state = last output)
Y0, Y1 = output_seqs

# if we were running 50 time steps -> not convenient to create 50 placeholders
# -> create a single placeholder of shape [None, n_steps, n_inputs] # 1st dim = batch_size
# then it extracts a list of input sequences for each time step

# X_seqs is a list of n_steps tensors of shape [None, n_inputs] # 1st dim = batch_size
# use transpose() to swap the first 2 dimensions so that the time steps are the 1st dim
# then unstack() to extract a list of tensors along the 1st dim (=1 tensor per time step)
# finally stack() to merge all output tensors into single tensor, 
# and swap the first 2 dim to get a final output tensor of shape [None, n_steps, n_neurons] # 1st dim = batch_size


X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
# swap to have time step as 1st dim and unstack to unstack to have 1 tensor per time step
X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2])) 
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0,X1], dtype=tf.float32)
# stack to merge all output tensors into one single and swap to have # of steps as 1st dim
outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])

# now we can run the RNN by feeding 1 single tensor containing all the mini-batch sequences:
X_batch = np.array([
# t=0         t=1
[[0,1,2], [9,8,7]],  # instance 0
[[3,4,5], [0,0,0]],  # instance 1
[[6,7,8], [6,5,4]],  # instance 2
[[9,0,1], [3,2,1]]   # instance 3
]) 
# X_batch.shape = (4,2,3)
with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X:X_batch})
# => get a single output_vals for all instances, all time steps and all neurons
    
# but this approach still builds a graph containing 1 cell per time step
# => very large graph ! 
# (memory issue: must store all the tensor values during forward pass to use them to compute gradients in reverse pass)
# => use dynamic_rnn()
    
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
# -> dynmic_rnn() uses a while_loop()
# -> use swap_memory=True to swap GPU's to CPU's memory (avoid OOM error)
# -> outputs a single tensor for all outputs at every time step, of shape [None, n_steps, n_neurons]
# (no need to stack, unstack, transpose)

# to handle variable input sequence length:
# set "sequence_length" parameter when calling dynamic_rnn() 
# -> 1D tensor indicating the length of the input sequence for each instance, e.g.
seq_length = tf.placeholder(tf.int32, [None])
# [...]
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)

# e.g.
X_batch = np.array([
    # step 0         step 1
    [[0,1,2], [9,8,7]],  # instance 0
    [[3,4,5], [0,0,0]],  # instance 1 (padded with a zero vector)
    [[6,7,8], [6,5,4]],  # instance 2
    [[9,0,1], [3,2,1]]   # instance 3
]) 
seq_length_batch = np.array([2,1,2,2])
# input tensor 2nd dimension is the size of the longest sequence
# -> the second instance contains only 1 sequence, must be padded with 0 vector !
# now feed values for both placeholders and seq length
with tf.Session() as sess:
    init.run()
    output_vals, states_val = sess.run([outputs, states], feed_dict={X:X_batch, seq_length:seq_length_batch})
# the RNN will output 0 vectors for every time step past the input sequence length
# (the 2nd instance for the 2nd time step will be 0)
# the states tensor contains the final state of each cell (excluding zero vectors)
# (corresponds to the output at t=1, except for the 2nd instance, will be t=0)
    
# for variable output sequence length ? if same as input sequence length, can be set using sequence_length
# otherwise, most common solution is to define a special output called "end-of-sequence token" (EOS token)
# -> any output past the EOS should be ignored (cf. later)
    
# TRAINING RNN: unroll through time and use backpropagation
# = backpropagation through time (BPTT)
# NB: gradients flow backward through all the outputs used by the cost function, not just throught the final output
# the same parameters W and b are used at each time step -> backpropagation will sum over all time steps
    
# example: train RNN to classify images (NB: for example purpose; CNN would be more appropriate)
# treat 28x28 pictures as sequences of 28 rows of 28 pixels
# use cells of 150 recurrent neurons
# + a fully connected containing 10 neurons (1 per class) connected to the output of last time step
# and followed by softmax activation

from tensorflow.contrib.layers import fully_connected
    
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
# the fully-connected layer takes final state (=28th output) as input (hold in the "states" tensor)
# recall: for basic cell, state = output
logits = fully_connected(states, n_outputs, activation_fn=None)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

# load the MNIST data, reshape data to [batch_size, n_steps, n_inputs]
from tensorflow.examples.tutorial.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")
X_test = mnist.test.images.reshape((-1,n_steps,n_inputs))
y_test = mnist.test.labels

# training similar to CNN, except reshaping each training batch before feeding it to the network
n_epochs = 100
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            # train (optimize) for the current mini-batch
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        # at the end of the epoch, accuracy on last batch data and full test
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:X_test, y:y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

# specify an initializer by wrapping its construction code in a variable scope, e.g.:
# variable_scope("rnn", initializer=variance_scaling_initializer()) 
# to use He initialization

# Train RNN on time series (predict the next value in a generated time series)
# e.g. each training instance is a randomly selected sequence of 20 consecutive values from the time series
# target sequence is the same as input sequence shifted by one time step
# create a RNN with 100 recurrent neurons unroll over 20 time steps since each training will be 20 inuts long
# each input contains 1 feature (the value at that time)
# targets are also sequences of 20 inputs, each containing only 1 value
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# NB: in general, more than 1 input feature (e.g. if you want to predict stock prices, you will also have
# other input features like ratings, prices of competing stock or any other features at each time step)

# at each time step, we have now as output a vector of size 100
# but we actually want a single value as output
# -> wrap the cell in an OutputProjectionWrapper
# (a cell wrapper acts like a normal cell but adds some functionality)
# OutputProjectionWrapper adds a fully connected layer of linear neurons (i.e. without any activation function)
# on the top of each output (but does not change the state of the cell)
# all the fully connected layers share the same trainable bias and weight terms
# wrapp the cell into the wrapper
cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
# use MSE as cost function and Adam optimizer
learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# initialize the variables
init = tf.global_variables_initializer()

# EXECUTION PHASE:
n_iterations = 10000
batch_size = 50
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = [...] # fetch the data
        sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print(iteration, "MSE:", mse)
            
# once the model is trained, make predictions
X_new = [...] # new sequences
y_pred = sess.run(outputs, feed_dict={X:X_new})

# OutputProjectionWrapper is the simplest but not the most efficient solution to reduce dimensionality
# more efficient way:
# -> reshape RNN outputs [batch_size, n_steps, n_neurons] to [batch_size*n_steps, n_neurons]
# -> apply fully-connected layer with output of appropriate size (1 in our example) 
# -> reshape FCL output [batch_size*n_steps, n_neurons] to [batch_size, n_steps, n_outputs]
# implementation:
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# reshape the outputs
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
# FC layer without activation function -> this is simply a projection
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
# unstack the output
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

# CREATIVE RNN: provide seed sequence with n_steps values (e.g. full of 0),
# use this model to predict the next value and append it to a sequence
# feed the last n_step values to the model to predict the next value, and so on
# -> generates a new sequence
sequence = [0.] * n_steps 
for iteration in range(300):
    # extract and reshape the last n_step values
    X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps,1) 
    y_pred = sess.run(outputs, feed_dict={X:X_batch})
    sequence.append(y_pred[0,-1,0])
    
# DEEP RNN = stack multiple layers of cells
# create several cells and stack them into MultiRNNCell
n_neurons = 100
n_layers = 3
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell]*n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
# => states is tuple containing 1 tensor per layer, each representing the final state of layer's cell
# with shape [batch_size, n_neurons]
# to have states as a single tensor concatenated along column axis of shape [batch_size, n_layers*n_neurons]
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32, state_is_tuple=False)

# deep RNN can be efficiently distributed by pinning each layer to a different GPU

# NB: BasicRNNCell() is cell factory, not actual cell
# the cells get created later, when dynamic_run() is run
# (call MultiRNNCell which calls BasicRNNCell which create the cells)

# if very deep RNN -> risk of overfitting
# apply dropout before or after RNN
# DropoutWrapper to apply dropout between RNN layers
keep_prob = 0.5
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep=keep_prob)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell_drop] * n_layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, dtype=tf.float32)
# -> problem with this code ! will apply dropout also during testing !
# (DropoutWrapper has no is_training parameter)
# -> either write a wrapper class 
# or have 2 different graphs, 1 for train and 1 for test, e.g.
import sys
is_training = (sys.argv[1] == "train")

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
if is_training:
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep=keep_prob)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
# [...] # build the rest of the graph
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    if is_training:
        init.run()
        for iteration in range(n_iterations):
            # train the model
        save_path = saver.save(sess, "/tmp/my_model.ckpt")
    else:
        saver.restore(sess, "/tmp/my_model.ckpt")
        # [...] use the model

# standard  tricks to alleviate the problems of vanishing/exploding gradients
# - good parameter initialization
# - non-saturating activation functions
# - batch normalization
# - gradient clipping
# - fast optimizer
# may not be enough for RNN
# - truncated backpropagation through time -> unroll RNN only a limited number of time steps
# (implemented simply by truncating input sequences)
# -> but will not be able to learn long-term patterns
# -> shortened sequences could hold both old and recent data

# besides long training time, 2nd of problem of RNNs is that memory of first inputs gradually fades away
# -> cells with long-term memory 

# LSTM cell
# -> converges faster
# -> detects long-term dependencies
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
# -> manages 2 state vectors (kept separately for performance reason)
# (short-term state and long-term state vectors)
# the network can learn what to store in the long-term state, what to throw away, what to read from it
# long-term state traverses the network
# -> input gate that selects some memories
# -> forget gate dropping some memories
# => adds the new memories to the memories (at each state, some memories are dropped and other added)
# -> output gate: after the addition, long-term state is copied and passed through tanh function and the result is filtered
# => this produces the short-term state (=cell's output at this time step)

# in details:
# - input vector and previous short-term memory state fed to 4 fully-connected layers
# -> 1 main layer with usual role of analyzing the current inputs and the previous short-term state
#    (in the basic cell, there is only this layer)
#    (in LSTM cell: does not get straight out, but partially stored in the long-term state)
# -> 3 other layers are gate controllers; use logistic activation function
#    (outputs fed to element-wise multiplication: if 0, they close the gate; if 1, they open it)
#     each connected to both input vector and previous short-term state (input)
#     each connected to a different gate (output)
#    => forget gate: controls which part of the long-term state should be erased
#    => input gate: controls which part should be added to the long-term state (hence "partially" stored here above)
#    => output gate: controls which part of the long-term state should be read and output at thsi time step

# => recognize an important input (role of the input gate), store it in long-term state,
# preserve it as long as needed (forget gate) and extract it whenever needed

# by default bias terms for the 4 layers are initialized with 1s instead of 0s 
# -> prevent forgetting everything at the beginning

# PEEPHOLE CONNECTIONS: in LSTM, gate controllers connected only to input connections and previous short-term state
# variant of LSTM to give more context -> peek at long-term state as well
# -> previous long-term state added as input to controllers of forget and input gate
# -> current long-term state added as input to controller of output gate
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, use_peepholes=True)

# GRU: other variant of LSTM
# simplified version of LSTM that seems to work as well
# -> both state vectors are merged into a single one
# -> a single controller controls both forget and input gate
# (if the gate controller outputs 1 -> input gate is open and forget gate is closed; the reverse if 0)
# ( = whenever a memory must be stored, the location where it will be stored will be erased first)
# -> no output gate, the full state vector is output at every time step
# -> but new controller that controls which parts of the previous state will be shown to main layer
gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)

# natural language processing
# instead of one-hot encoding, most common solution is to represent each word using a small and dense vector
# called an embedding (initialized randomly, and let the NN learns a good embedding during training)
# (similar words will gradually cluster together)
# in TensorFlow, first create the variable representing the embeddings for every word (random initialization)
vocabulary_size = 50000
embedding_size = 150
embedings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# to feed the network, preprocess the sentence and break it into words
# then you can look up each word's integer identifier in a dictionary
# -> feed the word identifiers using a placeholder
train_inputs = tf.placeholder(tf.float32, shape=[None])
# -> apply embedding_lookup() to get corresponding embeddings
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# once the model has learned the good embedding -> can be reused in any NLP application 
# pretrained embedding can be frozen (create embedding variables with trainable=False)
# or let backpropagation tweak them

# embeddings useful for representing categorical attributes

# ENCODER-DECODER NETWORK FOR MACHINE TRANSLATION
# English sentences fed to the encoder
# the decoder outputs the French translations
# French translations are also used as inputs to the decoder, but pushed back by 1 step
# (-> the decoder is given as input the word that it should have output at the previous step)
# usually sentences are reversed before feeding to the encoder -> ensures that the beginning of the 
# English sentence will be fed last (useful because that's generally the 1st thing the decoder needs to translate)

# - this is the word embeddings that are fed
# - at each step the decoder outputs a score for each word in the output vocabulary
# and then the softmax layer turns these scores into probabilities
# - the word with highest probability is output
# (similar to regular classification, max_cross_entropy_with_logits() can be used)

# for inference time: no target sequence, simply feed the decoder the word output at previous step