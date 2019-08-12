import tensorflow as tf

# high-level TensorFlow API

# sentiment RNN constructor (only build method shown)

class SentimentRNN(object):
    def __init__(self, n_words, seq_len=200, lstm_size=256, num_layers=1, batch_size=64, learning_rate=0.0001, embed_size=200):
        self.n_words = n_words
        self.seq_len = seq_len
        self.lstm_size = lstm_size # number of hidden units
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size
        
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            
    def build(self):
        tf_x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_len), name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=(self.batch_size), name='tf_y')
        tf_keepbrob = tf.placeholder(tf.float32, name='tf_keepprob')
        # create the embedding layer
        embedding = tf.Variable(tf.random_uniform(self.n_words, self.embed_size), minval=-1, maxval=1, name="embedding")
        embed_x = tf.nn.embedding_lookup(embedding, tf_x, name="embeded_x")
        # define LSTM cell and stack them together
        cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.lstm_size), output_keepproba=tf_keepprob) for i in range(self.num_layers)])
        # define the initial state
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        # output
        lstm_outputs, self.final_states = tf.nn.dynamic_rnn(cells, embed_x, initial_state=self.initial_state)        
        logits = tf.layers.dense(inputs=lstm_outputs[:,-1], units=1, activation=None, name="logits")
        logits = tf.squeeze(logits, name="logits_squeezed")
        y_proba = tf.nn.sigmoid(logits, name="probabilities")
        predictions = {'probabilities': y_proba, 'labels': tf.cast(tf.round(y_proba), tf.int32, name='labels')}
        # define cost function
        cost = tf.reduce_mean(tf.nn.simgoid_cross_entropy_with_logits(labels=tf_y, logits=logits), name='cost')
        # define optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')

# character-level language modeling constructor(only build method shown)

class CharCNN(object):
    def __init__(self, num_classes, batch_size=64, num_steps=100, lstm_size=128, num_layers=1, learning_rate=0.001, keep_prob=0.5, grad_clip=5, sampling=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build(sampling=sampling)
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            
    def build(self, sampling):
        if sampling:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = self.batch_size, self.num_steps
        tf_x = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')
        # one-hot encoding
        x_onehot = tf.one_hot(tf_x, depth = self.num_classes)
        y_onehot = tf.one_hot(tf_y, depth = self.num_classes)
        # build the multi-layer rnn cells
        cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(self.lstm_size), output_keep_prob=tf_keepprob) for _ in range(self.num_layers)])
        # define the initial state
        self.initial_state = cells.zero_state(batch_size, tf.float32)
        # run each sequence through RNN
        lstm_outputs, self.final_states = tf.nn.dynamic_rnn(cells, x_onehot, initial_state = self.initial_state)
        seq_output_reshaped = tf.reshape(lstm_outputs, shape=[-1, self.lstm_size], name="seq_output_reshaped")
        logits = tf.layers.dense(inputs=seq_output_reshaped, units=self.num_classes, activation=None, name="logits")
        proba = tf.nn.softmax(logits, name="probabilities")
        y_reshaped = tf.reshape(y_onehot, shape=[-1,self.num_classes], name="y_reshaped")
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped), name='cost')
        # gradients clipping to avoid exploding gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")
        
            
            