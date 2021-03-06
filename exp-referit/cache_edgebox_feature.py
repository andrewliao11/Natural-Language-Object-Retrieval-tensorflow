from __future__ import division, print_function

import sys
import numpy as np
import skimage.io
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
sys.path.append('./external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe
import cv2
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import util
from captioner import Captioner
import retriever

################################################################################
# Test Parameters

# Test on either all annotated regions, or top-100 EdgeBox proposals
# See Section 4.1 in the paper for details
candidate_regions = 'proposal_regions'
#candidate_regions = 'annotated_regions'

# Whether or not scene-level context are used in predictions
use_context = True
D = 1
bbox_dir = './data/referit_visualize/annotated/'

if use_context:
    lstm_net_proto = './prototxt/scrc_word_to_preds_full.prototxt'
    pretrained_weights_path = './models/scrc_full_vgg.caffemodel'
else:
    lstm_net_proto = './prototxt/scrc_word_to_preds_no_context.prototxt'
    pretrained_weights_path = './models/scrc_no_context_vgg.caffemodel'

gpu_id = 1  # the GPU to test the SCRC model
correct_IoU_threshold = 0.5

tst_imlist_file = './data/split/referit_test_imlist.txt'
################################################################################

image_dir = './datasets/ReferIt/ImageCLEF/images/'
proposal_dir = './data/referit_edgeboxes_top100/'
cached_context_features_dir = './data/referit_context_features/'

imcrop_dict_file = './data/metadata/referit_imcrop_dict.json'
imcrop_bbox_dict_file = './data/metadata/referit_imcrop_bbox_dict.json'
query_file = './data/metadata/referit_query_dict.json'
vocab_file = './data/vocabulary.txt'

# utilize the captioner module from LRCN
image_net_proto = './prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
captioner = Captioner(pretrained_weights_path, image_net_proto, lstm_net_proto,
                      vocab_file, gpu_id)
captioner.set_image_batch_size(50)
vocab_dict = retriever.build_vocab_dict_from_captioner(captioner)

# Load image and caption list
imlist = util.io.load_str_list(tst_imlist_file)
num_im = len(imlist)
query_dict = util.io.load_json(query_file)
imcrop_dict = util.io.load_json(imcrop_dict_file)
imcrop_bbox_dict = util.io.load_json(imcrop_bbox_dict_file)

# Load candidate regions (bounding boxes)
load_proposal = (candidate_regions == 'proposal_regions')
candidate_boxes_dict = {imname: None for imname in imlist}
for n_im in range(num_im):
    if n_im % 1000 == 0:
        print('loading candidate regions %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    # from edgebox
    if load_proposal:
        proposal_file_name = imname + '.txt'
        boxes = np.loadtxt(proposal_dir + proposal_file_name)
        boxes = boxes.astype(int).reshape((-1, 4))
    # from annotated bbox
    else:
        boxes = [imcrop_bbox_dict[imcrop_name]
                 for imcrop_name in imcrop_dict[imname]]
        boxes = np.array(boxes).astype(int).reshape((-1, 4))
    candidate_boxes_dict[imname] = boxes


################################################################################
# Test recall
K = 100  # evaluate recall at 1, 2, ..., K
topK_correct_num = np.zeros(K, dtype=np.float32)
total_num = 0
sample_im = num_im

for n_im in range(sample_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    # gt
    imcrop_names = imcrop_dict[imname]
    candidate_boxes = candidate_boxes_dict[imname]

    im = skimage.io.imread(image_dir + imname + '.jpg')
    imsize = np.array([im.shape[1], im.shape[0]])  # [width, height]

    # Compute local descriptors (local image feature + spatial feature)
    descriptors = retriever.compute_descriptors_edgebox(captioner, im,
                                                        candidate_boxes,'fc7') # (100,4096)
    spatial_feats = retriever.compute_spatial_feat(candidate_boxes, imsize) # (100,8)
    np.savez('./data/ReferIt/referit_proposal_feature/'+imname, spatial_feat=spatial_feats,local_feature=descriptors)


    # print intermediate results during testing
    if (n_im+1) % 1000 == 0:
        print('Recall on first %d test images' % (n_im+1))
        for k in [0, 10-1]:
            print('\trecall @ %d = %f' % (k+1, topK_correct_num[k]/total_num))

print('Final recall on the whole test set')
for k in [0, 10-1]:
    print('\trecall @ %d = %f' % (k+1, topK_correct_num[k]/total_num))
################################################################################
