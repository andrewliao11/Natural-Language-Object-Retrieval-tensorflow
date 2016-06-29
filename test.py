'''
Using gradient clipping
'''


from __future__ import print_function
# import caffe before tf, or segmentation fault
#from initialize import load_caffemodel
import sys
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
sys.path.append('./external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe
from captioner import Captioner
import tensorflow as tf
import numpy as np
import time
from tensorflow.models.rnn import rnn_cell
import retriever
import os, argparse
import pdb
import re
import util
import skimage.io
from keras.preprocessing import sequence

gpu_id = 0
sample_im = 100

# path
context_feature_path = './data/referit_context_features'
training_data_path = './data/training/50_training_data'
vocab_file = './data/vocabulary.txt'
test_imlist_path = './data/split/referit_test_imlist.txt'
test_imcrop_dict_path = './data/metadata/referit_imcrop_dict.json'
test_imcrop_bbox_dict_path = './data/metadata/referit_imcrop_bbox_dict.json'
test_query_path = './data/metadata/referit_query_dict.json'
image_dir = './datasets/ReferIt/ImageCLEF/images/'
proposal_dir = './data/referit_edgeboxes_top100/'
cached_context_features_dir = './data/referit_context_features/'


# Check point
save_checkpoint_every = 25000           # how often to save a model checkpoint?
test_model_path = './test_models/model-10'

# Train Parameter
dim_image = 4096
dim_hidden = 1000
n_epochs = 100
batch_size = 50
learning_rate = 0.001
MAX_QUERY_WORDS = 20+1
dim_coordinates = 8
max_grad_norm = 10

# test param
correct_IoU_threshold = 0.5
candidate_regions = 'proposal_regions'
K = 100

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
	self.lstm_lang = rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = True)
	self.lstm_context = rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = True)
	self.lstm_local = rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = True)

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

    def build_generator(self):
	
	# placeholder is for feeding data
	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])  # (batch_size, dim_image)
	local_image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
	query = tf.placeholder(tf.int32, [self.batch_size, MAX_QUERY_WORDS])
	query_mask = tf.placeholder(tf.float32, [self.batch_size, MAX_QUERY_WORDS])
	bbox = tf.placeholder(tf.float32, [self.batch_size, self.dim_coordinates])

	# [image] embed image feature to dim_hidden
        image_emb = tf.nn.bias_add(tf.matmul(image, self.embed_image_W), self.embed_image_b) # (batch_size, dim_hidden)
	local_image_emb = tf.nn.bias_add(tf.matmul(local_image, self.embed_local_W), self.embed_local_b) # (batch_size, dim_hidden)
	
        score = tf.zeros([self.batch_size], tf.float32)

	state_lang = tf.zeros([self.batch_size, self.lstm_lang.state_size])
	state_context = tf.zeros([self.batch_size, self.lstm_context.state_size])
	state_local = tf.zeros([self.batch_size, self.lstm_local.state_size])
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

            context_emb = tf.nn.xw_plus_b(context, self.W_context, self.B_context)
            local_emb = tf.nn.xw_plus_b(local, self.W_local, self.B_local)
            word_pred = tf.add(context_emb, local_emb)

	    max_prob_index = tf.argmax(word_pred, 1) # b

	    labels = tf.expand_dims(query[:,j], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            with tf.device('/cpu:0'):
                onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.dict_words]), 1.0, 0.0)
	    current_score = tf.mul(onehot_labels, word_pred)
	    current_score = tf.reduce_sum(current_score, 1)
	    current_score = tf.mul(current_score, query_mask[:,j])
	    current_score = tf.reshape(current_score, [1,self.batch_size])
	    current_score = tf.nn.softmax(current_score)
	    score = tf.add(score, current_score)

            with tf.device("/cpu:0"):
                tf.get_variable_scope().reuse_variables()
                query_emb = tf.nn.embedding_lookup(self.query_emb_W, max_prob_index)

	return score, image, local_image, query, query_mask, bbox

def test():

    print ('Building vocab dict')
    vocab_dict = retriever.build_vocab_dict_from_file(vocab_file)
    dict_words = len(vocab_dict.keys())
    f = open(test_imlist_path)
    test_imlist = f.read()
    test_imlist = re.split('\n',test_imlist)
    test_imlist.pop()
    num_im = len(test_imlist)

    print ('Building model')
    # batch size = 100(from edgebox)
    model = Answer_Generator(
            dim_image = dim_image,
            dict_words = dict_words,
	    dim_hidden = dim_hidden,
            batch_size = 100,
            drop_out_rate = 0,
	    dim_coordinates = dim_coordinates,
            bias_init_vector = None)
    
    tf_score, tf_image, tf_local_image, tf_query, tf_query_mask, tf_bbox = model.build_generator()
    print ('Building model successfully')
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, test_model_path)
 
    load_proposal = (candidate_regions == 'proposal_regions')
    candidate_boxes_dict = load_candidate(test_imlist, load_proposal)
    imcrop_dict = util.io.load_json(test_imcrop_dict_path)
    imcrop_bbox_dict = util.io.load_json(test_imcrop_bbox_dict_path)
    query_dict = util.io.load_json(test_query_path)
    # Test recall
    topK_correct_num = np.zeros(K, dtype=np.float32)
    total_num = 0

    tStart_total = time.time()

    for n_im in range(sample_im):

	print('Testing image %d / %d' % (n_im, num_im))
	imname = test_imlist[n_im]
	#print ('Load context feature')
	context_features = load_context_feature(imname)
	context_features = np.repeat(context_features,100,axis=0)
    	# gt
    	imcrop_names = imcrop_dict[imname]
    	candidate_boxes = candidate_boxes_dict[imname]

	im = skimage.io.imread(image_dir + imname + '.jpg')
    	imsize = np.array([im.shape[1], im.shape[0]])  # [width, height]
	local = np.load('./data/ReferIt/referit_proposal_feature/'+imname+'.npz')
	local_feature = local['local_feature']
	spatial_feats = local['spatial_feat']

	num_imcrop = len(imcrop_names)
	num_proposal = candidate_boxes.shape[0]
    	for n_imcrop in range(num_imcrop):
            imcrop_name = imcrop_names[n_imcrop]
	    if imcrop_name not in query_dict:
            	continue
            gt_bbox = np.array(imcrop_bbox_dict[imcrop_name])
            IoUs = retriever.compute_iou(candidate_boxes, gt_bbox)
            for n_sentence in range(len(query_dict[imcrop_name])):
            	sentence = query_dict[imcrop_name][n_sentence]
		sentence = np.tile(sentence, 100)
            	# Scores for each candidate region
		# TESTING
		current_query_ind = map(lambda cap: [vocab_dict[word] for word in cap.lower().split(' ') if word in vocab_dict], sentence)
            	current_query_matrix = sequence.pad_sequences(current_query_ind, padding='post', maxlen=MAX_QUERY_WORDS-1)
            	current_query_matrix = np.hstack( [current_query_matrix, np.zeros( [len(current_query_matrix),1]) ] ).astype(int)
           	current_query_mask = np.zeros((current_query_matrix.shape[0], current_query_matrix.shape[1]))
            	nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_query_matrix ))
            	for ind, row in enumerate(current_query_mask):
                    row[0:nonzeros[ind]] = 1
            	# do the testing process!!!
            	scores = sess.run([tf_score],
                    	feed_dict={
                            tf_local_image: local_feature,
                            tf_image: context_features,
                            tf_query: current_query_matrix,
                            tf_query_mask: current_query_mask,
                            tf_bbox: spatial_feats
                        })
		scores = scores[0][0]
		# Evaluate the correctness of top K predictions
            	topK_ids = np.argsort(-scores)[:K]
            	topK_IoUs = IoUs[topK_ids]
            	# whether the K-th (ranking from high to low) candidate is correct
            	topK_is_correct = np.zeros(K, dtype=bool)
            	topK_is_correct[:len(topK_ids)] = (topK_IoUs >= correct_IoU_threshold)
            	# whether at least one of the top K candidates is correct
            	topK_any_correct = (np.cumsum(topK_is_correct) > 0)
            	topK_correct_num += topK_any_correct
            	total_num += 1
	
	# print intermediate results during testing
    	if (n_im+1) % 1000 == 0:
            print('Recall on first %d test images' % (n_im+1))
            for k in [0, 10-1]:
            	print('\trecall @ %d = %f' % (k+1, topK_correct_num[k]/total_num))


    print('Final recall on the whole test set')
    for k in [0, 10-1]:
    	print('\trecall @ %d = %f' % (k+1, topK_correct_num[k]/total_num))


if __name__ == '__main__':
    with tf.device('/gpu:'+str(gpu_id)):
        test()
