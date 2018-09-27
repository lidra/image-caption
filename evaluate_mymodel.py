
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io
import cPickle
import configuration

class Evaluate(object):

  def __init__(self):

        x = cPickle.load(open("../data/mscoco/data.p","rb"))
	train, val, test = x[0], x[1], x[2]
	wordtoix, ixtoword = x[3], x[4]
	del x
	n_words = len(ixtoword)
	    
	x = cPickle.load(open("../data/mscoco/word2vec.p","rb"))
	W = x[0]
	del x
	data = scipy.io.loadmat('../data/mscoco/resnet_feats.mat')
	img_feats = data['feats'].astype(float)
        print("finish loading data")
        self.val = test #test the test images
        self.img_feats = img_feats
	self.ixtoword = ixtoword
      
  def evaluate(self):
        g = tf.Graph()
	with g.as_default():
            model_config = configuration.ModelConfig()
	    training_config = configuration.TrainingConfig()

	    #initializer method
	    initializer = tf.random_uniform_initializer(
		minval=-model_config.initializer_scale,
		maxval=model_config.initializer_scale)

	    seq_embeddings = None
	    image_feed = tf.placeholder(dtype=tf.float32, shape=[2048], name="image_feed")
	    input_feed = tf.placeholder(dtype=tf.int32,
			                  shape=[None],  # batch_size
			                  name="input_feed")

	    # Process image and insert batch dimensions.
	    image_fea = tf.expand_dims(image_feed, 0)
	    input_seqs = tf.expand_dims(input_feed, 0)

	    with tf.variable_scope("seq_embedding"), tf.device("/gpu:0"):
	      embedding_map = tf.get_variable(
		  name="map",
		  shape=[model_config.vocab_size, model_config.embedding_size],
		  initializer=initializer)
	    seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

	    with tf.variable_scope("image_embedding") as scope:
	      image_embeddings = tf.contrib.layers.fully_connected(
		  inputs=image_fea,
		  num_outputs=model_config.embedding_size,
		  activation_fn=None,
		  weights_initializer=initializer,
		  biases_initializer=None,
		  scope=scope)

	    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
		num_units=model_config.num_lstm_units, state_is_tuple=True) 

	    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
	      # Feed the image embeddings to set the initial LSTM state.
	      zero_state = lstm_cell.zero_state(
		  batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
	      _, initial_state = lstm_cell(image_embeddings, zero_state)

	      # Allow the LSTM variables to be reused.
	      lstm_scope.reuse_variables()

	      # In inference mode, use concatenated states for convenient feeding and
	      # fetching.
	      tf.concat(axis=1, values=initial_state, name="initial_state")

	      # Placeholder for feeding a batch of concatenated states.
	      state_feed = tf.placeholder(dtype=tf.float32,
		                            shape=[None, sum(lstm_cell.state_size)],
		                            name="state_feed")
	      state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

	      # Run a single LSTM step.
	      lstm_outputs, state_tuple = lstm_cell(
		    inputs=tf.squeeze(seq_embeddings, axis=[1]),
		    state=state_tuple)

	      # Concatentate the resulting state.
	      tf.concat(axis=1, values=state_tuple, name="state")
 

	    # Stack batches vertically.
	    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

	    with tf.variable_scope("logits") as logits_scope:
	      logits = tf.contrib.layers.fully_connected(
		  inputs=lstm_outputs,
		  num_outputs=model_config.vocab_size,
		  activation_fn=None,
		  weights_initializer=initializer,
		  scope=logits_scope)

	   
	    tf.nn.softmax(logits, name="softmax")

	    global_step = tf.Variable(
		initial_value=0,
		name="global_step",
		trainable=False,
		collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
            # Set up the Saver for saving and restoring model checkpoints.
            saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
            g.as_default()
            sess = tf.Session(graph=g)
            #load the trained model 
            with sess.as_default():
                saver.restore(sess, "log/model.ckpt") 

        print("finish initialization")

        x= self.val[0]
        lengths = [len(s) for s in x]
	n_samples = len(x)
	maxlen = np.max(lengths)
        #remove duplicate. Because one image has many captions.
        val_re = []
        for i in range(n_samples):
        	if self.val[1][i] not in val_re:
   			val_re.append(self.val[1][i])
        n_samples = len(val_re)
        print("n_samples:"+str(n_samples)+"maxlen:"+str(maxlen))
	z = np.array([self.img_feats[:,val_re[t]]for t in range(n_samples)])
        cap = np.zeros(( n_samples,maxlen))

        #generate captions.feed word one by one to the model.Start with 6800('#').Stop when get 0('.')
        for num in range(n_samples):
                if num%1000==0:
			print(num)
		initial_state = sess.run(fetches="lstm/initial_state:0",
                             feed_dict={"image_feed:0": z[num]})
                input_feed = np.array([6800])
                state_feed = initial_state
        	for s in range(maxlen):		
			softmax_output, state_output = sess.run(
				fetches=["softmax:0", "lstm/state:0"],
				feed_dict={
				    "input_feed:0": input_feed,
				    "lstm/state_feed:0": state_feed,
				})
                        softmax_output = softmax_output.reshape(softmax_output.shape[1])
			input_feed = [np.argsort(-softmax_output)[0]]
			#print(softmax_output.shape)
			#print(input_feed)
                        state_feed  = state_output
                        cap[num][s] = input_feed[0]
                        if input_feed[0]==0:
                                #print(cap[num])
                                break
			
        #get the real word by index
	precaptext=[]
	for i in range(n_samples):
		temcap=[]
		for j in range(maxlen):
			if cap[i][j]!=0:
				temcap.append(self.ixtoword[cap[i][j]])
			else:
				break
		precaptext.append(" ".join(temcap))
        #save the results to 'coco_5k_test.txt'
	print('write generated captions into a text file...')
	open('./coco_5k_test.txt', 'w').write('\n'.join(precaptext))
	
				



