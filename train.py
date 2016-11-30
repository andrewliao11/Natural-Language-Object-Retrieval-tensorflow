'''
1. Use hidden state to represent the lstm, instead of the output
2. add pretrained params
3. add bias term when predicting answer

4. use momentum optimizer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
#from tensorflow.models.rnn import rnn_cell
import retriever
import os
import pdb
import re
from keras.preprocessing import sequence

# path
context_feature_path = './data/referit_context_features'
training_data_path = './data/training/50_training_data'
vocab_file = './data/vocabulary.txt'
pretrain_model_path = './premodel.npz'

# Check point
save_checkpoint_every = 25000           # how often to save a model checkpoint?
model_path = './test_models'

# Train Parameter
dim_image = 4096
dim_hidden = 1000
n_epochs = 80
batch_size = 50
base_learning_rate = 0.001
MAX_QUERY_WORDS = 20+1
dim_coordinates = 8
max_grad_norm = 10

pretrained_weights = ['lstm_context/LSTMCell/W_0:0', 'lstm_context/LSTMCell/B:0', 'lstm_local/LSTMCell/W_0:0', 
				'lstm_local/LSTMCell/B:0','embed_local_W:0', 'embed_local_b:0',
				'embed_image_W:0', 'embed_image_b:0','W_local:0', 'B_local:0', 'W_image:0', 'B_image:0']

class Answer_Generator():
    def __init__(self, dim_image, dict_words, dim_hidden, batch_size, drop_out_rate, dim_coordinates, bias_init_vector=None):
        print('Initialize the model')
	self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.drop_out_rate = drop_out_rate
	self.dict_words = dict_words
	self.dim_coordinates = dim_coordinates

	# LSTM cell
	self.lstm_lang = tf.nn.rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = True,state_is_tuple=False)
        #self.lstm_lang_dropout = tf.nn.rnn_cell.DropoutWrapper(self.lstm_lang,output_keep_prob = 1-self.drop_out_rate)
	self.lstm_context = tf.nn.rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = True,state_is_tuple=False)
        #self.lstm_context_dropout = tf.nn.rnn_cell.DropoutWrapper(self.lstm_context,output_keep_prob = 1-self.drop_out_rate)
	self.lstm_local = tf.nn.rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = True,state_is_tuple=False)
        #self.lstm_local_dropout = tf.nn.rnn_cell.DropoutWrapper(self.lstm_local,output_keep_prob = 1-self.drop_out_rate)

	# image feature embedded
	self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.1,0.1), name='embed_image_W')
	if bias_init_vector is not None:
            self.embed_image_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_image_b')
        else:
            self.embed_image_b = tf.Variable(tf.zeros([self.dim_hidden]), name='embed_image_b')

        # local image feature embedded
        self.embed_local_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.1,0.1), name='embed_local_W')
        if bias_init_vector is not None:
            self.embed_local_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_local_b')
        else:
            self.embed_local_b = tf.Variable(tf.zeros([self.dim_hidden]), name='embed_local_b')
	
	# embed the word into lower space
	with tf.device("/cpu:0"):
            self.query_emb_W = tf.Variable(tf.random_uniform([dict_words, self.dim_hidden], -0.1, 0.1), name='query_emb_W')

	# embed lower space into answer
	self.W_context = tf.Variable(tf.random_uniform([self.dim_hidden, dict_words], -0.1, 0.1), name='W_context')
        self.W_local = tf.Variable(tf.random_uniform([self.dim_hidden, dict_words], -0.1, 0.1), name='W_local')
	self.B_context = tf.Variable(tf.random_uniform([dict_words], -0.1, 0.1), name='B_context')
        self.B_local = tf.Variable(tf.random_uniform([dict_words], -0.1, 0.1), name='B_local')
	#self.b_word_pred = tf.Variable(tf.zeros([dict_words]), name='b_word_pred')
	

    def build_model(self):
	
	# placeholder is for feeding data
	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
	local_image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
	query = tf.placeholder(tf.int32, [self.batch_size, MAX_QUERY_WORDS])
	query_mask = tf.placeholder(tf.float32, [self.batch_size, MAX_QUERY_WORDS])
	bbox = tf.placeholder(tf.float32, [self.batch_size, self.dim_coordinates])

	# [image] embed image feature to dim_hidden
	image_emb = tf.nn.bias_add(tf.matmul(image, self.embed_image_W), self.embed_image_b) # (batch_size, dim_hidden)
        local_image_emb = tf.nn.bias_add(tf.matmul(local_image, self.embed_local_W), self.embed_local_b) # (batch_size, dim_hidden)	
        loss = 0.0

	state_lang = tf.zeros([self.batch_size, self.lstm_lang.state_size])
	state_context = tf.zeros([self.batch_size, self.lstm_context.state_size])
	state_local = tf.zeros([self.batch_size, self.lstm_local.state_size])
	#state_langS = []
	#output_langS = []
	query_emb = tf.zeros([self.batch_size, self.dim_hidden])
	for j in range(MAX_QUERY_WORDS): 
	    
	    # language lstm
	    with tf.variable_scope("lstm_lang"):
                output_lang, state_lang = self.lstm_lang(query_emb, state_lang)
	    lang = tf.slice(state_lang, [0,0], [self.batch_size, self.dim_hidden])
	    # context lstm
	    
	    with tf.variable_scope("lstm_context"):
                output_context, state_context = self.lstm_context(tf.concat(1,[image_emb, lang]), state_context)
	    context = tf.slice(state_context, [0,0], [self.batch_size, self.dim_hidden])

            # local lstm
            with tf.variable_scope("lstm_local"):
                output_local, state_local = self.lstm_local(tf.concat(1,[local_image_emb, lang, bbox]), state_local)
            local = tf.slice(state_local, [0,0], [self.batch_size, self.dim_hidden])

	    #context_emb = tf.matmul(context, self.W_context)
	    #pdb.set_trace()
	    context_emb = tf.nn.xw_plus_b(context, self.W_context, self.B_context)
	    local_emb = tf.nn.xw_plus_b(local, self.W_local, self.B_local)
	    #local_emb = tf.matmul(local, self.W_local)
	    word_pred = tf.add(context_emb, local_emb)
	    #word_pred = tf.nn.bias_add(word_pred, self.b_word_pred)

	    labels = tf.expand_dims(query[:,j], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            with tf.device('/cpu:0'):
                onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.dict_words]), 1.0, 0.0)
	    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_pred,onehot_labels) # (batch_size, )
	    cross_entropy = tf.mul(cross_entropy, query_mask[:,j])
	    current_loss = tf.reduce_sum(cross_entropy)
	    loss += current_loss

	    with tf.device("/cpu:0"):
                tf.get_variable_scope().reuse_variables()
                query_emb = tf.nn.embedding_lookup(self.query_emb_W, query[:,j])

	loss = loss / tf.reduce_sum(query_mask)
	
	param = []
	for v in tf.all_variables():
	    if v.name in pretrained_weights:
		param.append(v)
	

	return loss, image, local_image, query, query_mask, bbox, param
	
def train():

    print ('Building vocab dict')
    vocab_dict = retriever.build_vocab_dict_from_file(vocab_file)
    data_list = os.listdir(training_data_path)
    num_batch = len(data_list)
    data_list = np.asarray(data_list)
    dict_words = len(vocab_dict.keys())

    print ('Building model')
    model = Answer_Generator(
            dim_image = dim_image,
            dict_words = dict_words,
	    dim_hidden = dim_hidden,
            batch_size = batch_size,
            drop_out_rate = 0.5,
	    dim_coordinates = dim_coordinates,
            bias_init_vector = None)
    
    tf_loss, tf_image, tf_local_image, tf_query, tf_query_mask, tf_bbox, tf_param = model.build_model()
    print ('Building model successfully')
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    writer = tf.train.SummaryWriter('./tf_log', sess.graph_def)
    saver = tf.train.Saver(max_to_keep=100)
    # gradient clipping 
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(tf_loss, tvars), max_grad_norm)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Adam Optimizer
    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(
      base_learning_rate,                # Base learning rate.
      global_step,  # Current index into the dataset.
      10000,          # Decay step.
      0.8,                # Decay rate.
      staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    #train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()
       
    #print ('Load pretrained model')
    #pretrain = np.load(pretrain_model_path) 
    #for row in tf_param:
    #    assign_op = row.assign(pretrain[re.split(':', row.name)[0].replace('/','8')])
    #    sess.run(assign_op)

    tStart_total = time.time()
    for epoch in range(n_epochs):
	# shuffle the training data
	index = np.arange(num_batch)
        np.random.shuffle(index)
	data_list = data_list[index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(num_batch)
	for current_batch_idx in range(num_batch):
            tStart = time.time()
	    current_data = np.load(os.path.join(training_data_path,data_list[current_batch_idx]))
	    current_query = current_data['raw_query']
            current_context = current_data['fc7_context']
            current_local = current_data['fc7_local']
            current_bbox_coordinates = current_data['bbox_coordinates']
            current_query_ind = map(lambda cap: [vocab_dict[word] for word in cap.lower().split(' ') if word in vocab_dict], current_query)
	    current_query_matrix = sequence.pad_sequences(current_query_ind, padding='post', maxlen=MAX_QUERY_WORDS-1)
	    current_query_matrix = np.hstack( [current_query_matrix, np.zeros( [len(current_query_matrix),1]) ] ).astype(int)
            current_query_mask = np.zeros((current_query_matrix.shape[0], current_query_matrix.shape[1]))
	    nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_query_matrix ))
	    for ind, row in enumerate(current_query_mask):
                row[0:nonzeros[ind]] = 1
            # do the training process!!!
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_local_image: current_local,
			tf_image: current_context,
                        tf_query: current_query_matrix,
			tf_query_mask: current_query_mask,
			tf_bbox: current_bbox_coordinates
                        })
	    loss_epoch[current_batch_idx] = loss
            tStop = time.time()
	    print ("Current learning rate:", learning_rate.eval(), "Global step:", global_step.eval())
            print ("Epoch:", epoch, ", Batch:", current_batch_idx, ", Loss=", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")
	# every 10 epoch: print result
	if np.mod(epoch, 5) == 0:
            print ("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
	print ("Epoch:", epoch, " done. Loss:", np.mean(loss_epoch))
        tStop_epoch = time.time()
        print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
    
    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")	


if __name__ == '__main__':
    with tf.device('/gpu:'+str(3)):
        train()
