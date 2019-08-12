# naive implementation of a basic conv1d
import numpy as np
def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
    res = []
    for i in range(0, int(len(x)/s), s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
    return np.array(res)

x = [1,3,2,4,5,6,1,3]
w  = [1,0,3,1,2] 
conv1d(x, w, p=2, s=1)   
    
    
# naive implementation of a basic conv2d
import numpy as np
def conv2d(X, W, p=(0,0), s=(1,1)):
    W_rot = np.array(W)[::-1,::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]
    X_padded = np.zeros(shape=(n1,n2))
    X_padded[p[0]:p[0]+X_orig.shape[0],
             p[1]:p[1]+X_orig.shape[1]] = X_orig
    res = []
    for i in range(0, int((X_padded.shape[0]-W_rot.shape[0])/s[0]) + 1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1]-W_rot.shape[1])/s[1]) + 1, s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0], j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub*W_rot))
    return np.array(res)
    
X = [[1,3,2,4], [5,6,1,3], [1,2,0,2], [3,4,3,2]]
W  = [[1,0,3], [1,2,1],[0,1,1]]
conv2d(X, W, p=(1,1), s=(1,1))
    
# LOADING AND PRE-PROCESSING DATA
# [...]
# function for iterating through mini-batches of data
def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arrange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])
        
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train-mean_vals)/std_val
X_valid_centered = (X_valid-mean_vals)/std_val
X_test_centered = (X_test-mean_vals)/std_val

############################
# CNN implementation in TensorFlow low-level API
############################
import tensorflow as tf
import numpy as np

# wrapper function for the convolution layers
def conv_layer(input_tensor, name, kernel_size, n_output_channels, padding_mode="SAME", strides=(1,1,1,1)):
    with tf.variable_scope(name):
        # get input channels: input tensor shape [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases, name="net_pre-activation")
        print(conv)
        conv = tf.nn.relu(conv, name="activation")
        print(conv)
        return(conv)
        
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    conv_layer(x, name="convtest", kernel_size=(3,3), n_output_channels=32)
del g,x

# wrapper function for the fully connected layers
def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope("name"):
        input_shape  = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
            weights_shape = [n_input_units, n_output_units]
            weights = tf.get_variable(name="_weights", shape=weights_shape)
            print(weights)
            biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_units]))
            print(biases)
            layer = tf.matmul(input_tensor, weights)
            print(layer)
            layer = tf.nn.bias_add(layer, biases, name="net_pre-activation")
            print(layer)
            if activation_fn is None:
                return layer
            layer = activation_fn(layer, name='activation')
            print(layer)
            return(layer)
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    fc_layer(x, name="fctest", n_output_units=32, activation_fn=tf.nn.relu)
del g,x

# use the wrapper functions to build the network
def build_cnn():
    tf_x = tf.placeholder(tf.float32, shape=[None, 784], name="tf_x")
    tf_y = tf.placeholder(tf.int32, shape=[None], name="tf_y")
    # reshape to 4d tensor
    tf_x_image = tf.reshape(tf_x, shape=[-1,28,28,1], name="tf_x_reshaped")
    # one-shot encoding
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name="tf_y_onehot")
    # 1st layer conv1
    h1 = conv_layer(tf_x_image, name="conv1", kernel_size=(5,5), padding_mode="valid", n_output_channels=32)
    # max pooling
    h1_pool = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="same")
    # 2nd layer conv2
    h2 = conv_layer(h1_pool, name="conv2", kernel_size=(5,5), padding_mode="valid", n_output_channels=64)
    # max pool
    h2_pool = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="same")
    # 3d layer, fully-connected
    h3 = fc_layer(h2_pool, name='fc3', n_output_channels=1024, active_fn=tf.nn.relu)
    # dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name="dropout-layer")
    # 4th layer: fully connected (linear activation)
    h4 = fc_layer(h3_drop, name="fc4", n_output_units=10, activation_fn=None)
    
    # predictions
    predictions={'probabilities': tf.nn.softmax(h4, name='probabilities'),
                 'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32, name="labels")}
                 
    # loss function 
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot), name="cross-entropy-loss")
    
    # optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name="train-op")
    
    # compute prediction acuracy
    correct_predictions = tf.equal(prediction['labels'], tf_y, name="correct_preds")
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='acurracy')


# 4 additional definitions to help
def save(saver, sess, epoch, path="./model/"):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)
    
def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-%d' % epoch))
    
def train(sess, training_set, validation_set = None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []    
    # initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())
    np.random.seed(random_seed) # for shuffling batch_generator
    for epoch in range(1, epochs+1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0':batch_x,'tf_y:0':batch_y,'fc_keep_prob:0':dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss += loss
        training_loss.append(avg_loss/(i+1))
        print("Epoch %02d Training Avg. Loss: %7.3f" % (epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0], 'tf_y:0':validation_set[1],'fc_keep_prob:0':1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print('Validation Acc: %7.3f' % valid_acc)
        else:
            print("")

def predict(sess, X_test, return_proba=False):
    feed = {"tf_x:0":X_test, 'fc_keep_prob:0':1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)

# create the TensorFlow graph
# define hyperparameters
learning_rate = 1e-4
random_seed = 123
# create the graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    # build the graph
    build_cnn()
    # saver
    saver = tf.train.Saver()
    # create the session and train the model
    with tf.Session() as sess:
        train(sess, training_set=(X_train_centered, y_train), validation_set=(X_valid_centered, y_valid), initialize=True, random_seed=123)
        save(saver, sess, epoch=20)
# calculate accuracy on test set restoring the saved model
del g
# create new grpah and build the model
g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    # build the graph
    build_cnn()
    # saver
    saver = tf.train.Saver()
# create a new session and restore the model
    with tf.Session(graph=g2) as sess:
        load(saver, sess, epoch=20, path='./model')
        preds = predict(sess, X_test_centered, return_proba=False)
        print('Test accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))

# look at predicted labels
np.set_printoptions(precision=2, suppress=True)

with tf.Session(graph=g2) as sess:
    load(saver, sess, epoch=20, path='./model/')
    print(predict(sess, X_test_centered[:10], return_proba=False))
    print(predict(sess, X_test_centered[:10], return_proba=True))


############################
# CNN implementation in TensorFlow high-level API
############################
