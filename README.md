# Natural-Language-Object-Retrieval_tf
Implement [Natural Language Object Retrieval](http://arxiv.org/abs/1511.04164) in Tensorflow. The original code is release [here](https://github.com/ronghanghu/natural-language-object-retrieval), written in Caffe.   
note that some of the codes are from [natural-language-object-retrieval](https://github.com/ronghanghu/natural-language-object-retrieval)

## What is natural language object retrieval?
In real AI, you may refer something with its location, color, or other charateritic. The AI robot should know where the corresponding object is. Here's the illustration:   
![ill](https://github.com/andrewliao11/Natural-Language-Object-Retrieval_tf/blob/master/img/illustration.png?raw=true)   
The blue box represents the ground truth, while the yellow stands for positive recall, red for negative recall.
   
   
## Requirement   
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Caffe](https://github.com/BVLC/caffe)


## Training
Here, all the experiments are done on [ReferIt](http://tamaraberg.com/referitgame/) dataset. You can also train on other dataset in the same way.
### Install Caffe
1. Download caffe: ```./external/download_caffe.sh```
2. modify Makefile.config
3. ```make -j```
4. ```make pycaffe```

### Preprocess 
1. Download the pretrained models: ```./models/download_trained_models.sh```
2. Download the ReferIt dataset: ```./datasets/ReferIt/ReferitData/download_referit_dataset.sh``` and ```./datasets/ReferIt/ImageCLEF/download_data.sh```
3. Download pre-extracted EdgeBox proposals: ```./data/download_edgebox_proposals.sh```
4. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: ```python ./exp-referit/preprocess_dataset.py```
5. Cache the scene-level contextual features to disk: ```python ./exp-referit/cache_referit_context_features.py```
6. Cache the bbox features to disk: ```python ./exp-referit/cache_referit_local_features.py```
7. Build training batches: ```python ./exp-referit/cache_referit_training_batches.py```

### Start training
Once you prepare the data, you can run the training code ```python train.py```

## Testing
Before testing, I cache the proposal feature to disk due to the RAM contraints.
1. cache the proposal feature: ```python ./exp-referit/cache_edgebox_feature.py```
2. choose which model you want to test by modifying [this line](https://github.com/andrewliao11/Natural-Language-Object-Retrieval_tf/blob/master/test.py#L44).
3. ```python test.py```
4. 
## Experiments result
 I use R@10 to evaluate the performance and all the test is on **[ReferIt](http://tamaraberg.com/referitgame/) dataset**.   
the performance through different epoch [without pretrained]:   
![](https://github.com/andrewliao11/Natural-Language-Object-Retrieval_tf/blob/master/img/wo_pretrained.png?raw=true)   
the performance through different epoch [with pretrained]:    
![](https://github.com/andrewliao11/Natural-Language-Object-Retrieval_tf/blob/master/img/w_pretrained.png?raw=true)   
And here is the loss thourgh every epoch:   
![](https://github.com/andrewliao11/Natural-Language-Object-Retrieval_tf/blob/master/img/loss.png?raw=true)

