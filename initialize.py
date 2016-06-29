from __future__ import division, print_function

import sys
import numpy as np
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
import caffe
import pdb

def load_caffemodel():

    pretrained_weights = {}
    old_prototxt = './prototxt/coco_pretrained.prototxt'
    old_caffemodel = './models/coco_pretrained_iter_100000.caffemodel'
    old_net = caffe.Net(old_prototxt, old_caffemodel, caffe.TRAIN)
    new_prototxt = './prototxt/scrc_full_vgg_buffer_50.prototxt'
    new_caffemodel = './exp-referit/caffemodel/scrc_full_vgg_init.caffemodel'
    new_net = caffe.Net(new_prototxt, new_caffemodel, caffe.TRAIN)

    pretrained_weights['embed_local_W'] = np.transpose(old_net.params['fc8'][0].data[...]) # 4096 1000
    pretrained_weights['embed_local_b'] = old_net.params['fc8'][1].data[...] # 1000
    pretrained_weights['embed_image_W'] = np.transpose(old_net.params['fc8'][0].data[...]) # 4096 1000
    pretrained_weights['embed_image_b'] = old_net.params['fc8'][1].data[...] # 1000

    # image lstm
    pretrained_weights['lstm_context_W'] = np.transpose(np.concatenate((old_net.params['lstm2'][2].data[...],
								old_net.params['lstm2'][0].data[...],
								old_net.params['lstm2'][3].data[...]),axis=1)) # 3000 4000
    pretrained_weights['lstm_context_B'] = old_net.params['lstm2'][1].data[...] # 4000
    pretrained_weights['lstm_local_W'] = np.transpose(np.concatenate((old_net.params['lstm2'][2].data[...],np.zeros([4000,8]),
                                                                old_net.params['lstm2'][0].data[...],
                                                                old_net.params['lstm2'][3].data[...]),axis=1)) # 3008 4000
    pretrained_weights['lstm_local_B'] = old_net.params['lstm2'][1].data[...]

    pretrained_weights['W_context'] = np.transpose(new_net.params['predict'][0].data[...])
    pretrained_weights['B_context'] = new_net.params['predict'][1].data[...]
    pretrained_weights['W_local'] = np.transpose(new_net.params['predict'][0].data[...])
    pretrained_weights['B_local'] = new_net.params['predict'][1].data[...]

    # use '8', since key can't contain '/' 
    np.savez('premodel', embed_image_W=pretrained_weights['embed_image_W'],
			embed_image_b=pretrained_weights['embed_image_b'],
			embed_local_W=pretrained_weights['embed_local_W'],
                        embed_local_b=pretrained_weights['embed_local_b'],
			lstm_context8LSTMCell8W_0=pretrained_weights['lstm_context_W'],
			lstm_context8LSTMCell8B=pretrained_weights['lstm_context_B'],
                        lstm_local8LSTMCell8W_0=pretrained_weights['lstm_local_W'],
                        lstm_local8LSTMCell8B=pretrained_weights['lstm_local_B'],
			W_context=pretrained_weights['W_context'],
			B_context=pretrained_weights['B_context'],
                        W_local=pretrained_weights['W_local'],
                        B_local=pretrained_weights['B_local'])

    return pretrained_weights

load_caffemodel()
