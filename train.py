"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io
import cPickle
import configuration


def main(unused_argv):
    # load data disk
    x = cPickle.load(open("./data/mscoco/data.p", "rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)

    x = cPickle.load(open("./data/mscoco/word2vec.p", "rb"))
    W = x[0]
    del x
    data = scipy.io.loadmat('./data/mscoco/resnet_feats.mat')
    img_feats = data['feats'].astype(float)
    print("finish loading data")

    g = tf.Graph()
    with g.as_default():
        # creat config objects which contain model and training configs
        model_config = configuration.ModelConfig()
        training_config = configuration.TrainingConfig()

        # initializer method
        initializer = tf.random_uniform_initializer(
            minval=-model_config.initializer_scale,
            maxval=model_config.initializer_scale)

        batch_size = model_config.batch_size  # batch_size = 32
        image_fea = tf.placeholder(tf.float32, shape=[None, 2048])
        input_seqs = tf.placeholder(tf.int32, shape=[None, None])
        target_seqs = tf.placeholder(tf.int32, shape=[None, None])
        input_mask = tf.placeholder(tf.int32, shape=[None, None])
        # creat the seq embedding map. It is random init.
        with tf.variable_scope("seq_embedding"), tf.device("/gpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[model_config.vocab_size, model_config.embedding_size],
                initializer=initializer)
        seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)
        # creat image embedding layer. It is just fully connected layer.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=image_fea,
                num_outputs=model_config.embedding_size,
                activation_fn=None,
                weights_initializer=initializer,
                biases_initializer=None,
                scope=scope)
        # creat lstm cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=model_config.num_lstm_units, state_is_tuple=True)
        # add dropout in training module. It will be removed in testing model
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=model_config.lstm_dropout_keep_prob,
            output_keep_prob=model_config.lstm_dropout_keep_prob)

        with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
            # Feed the image embeddings to set the initial LSTM state.
            zero_state = lstm_cell.zero_state(
                batch_size=batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(image_embeddings, zero_state)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()

            # Run the batch of sequence embeddings through the LSTM.
            sequence_length = tf.reduce_sum(input_mask, 1)
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=seq_embeddings,
                                                sequence_length=sequence_length,
                                                initial_state=initial_state,
                                                dtype=tf.float32,
                                                scope=lstm_scope)
            # lstm_outputs = tf.Print(lstm_outputs, [lstm_outputs, tf.shape(lstm_outputs), "anything I want"])

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=model_config.vocab_size,
                activation_fn=None,
                weights_initializer=initializer,
                scope=logits_scope)

        targets = tf.reshape(target_seqs, [-1])
        weights = tf.to_float(tf.reshape(input_mask, [-1]))

        # Compute losses.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                logits=logits)
        batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                            tf.reduce_sum(weights),
                            name="batch_loss")
        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        # Add summaries.
        tf.summary.scalar("losses/batch_loss", batch_loss)
        tf.summary.scalar("losses/total_loss", total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

        # get the steps
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        # learing rate
        learning_rate_decay_fn = None
        learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_r = _learning_rate_decay_fn(learning_rate, global_step)

        # Set up the training ops.
        # We change the learing_rate directly here rather than using learning_rate_decay_fn
        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=learning_r,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=None)  # learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
        print("finish building network")
        g.as_default()
        sess = tf.Session(graph=g)
        # init = tf.global_variables_initializer()
        with sess.as_default():
            tf.global_variables_initializer().run()
        print("finish initialization")

    # prepare the data.
    # add a 6880('#') before the input seqs
    def prepare_data(seqs):
        # x: a list of sentences
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        outputs = np.zeros((n_samples, maxlen)).astype('int64')
        x_mask = np.zeros((n_samples, maxlen)).astype(float)
        for idx, s in enumerate(seqs):
            inputs[idx, 0] = 6880
            inputs[idx, 1:lengths[idx]] = s[:lengths[idx] - 1]
            outputs[idx, :lengths[idx]] = s[:lengths[idx]]
            x_mask[idx, :lengths[idx]] = 1.
        return inputs, x_mask, outputs

    # generate data index by batches. It can shuffle data at the same time
    def get_minibatches_idx(n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
            minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    kf = get_minibatches_idx(len(val[0]), batch_size, shuffle=True)

    max_epochs = 57  # 56.46 for 1000000 steps

    for eidx in xrange(max_epochs):
        print("the " + str(eidx) + " epochs")
        kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

        saver_path = saver.save(sess, "log/model.ckpt")  # save/model.ckpt
        print("Model saved in file:", saver_path)
        for steps, train_index in kf:

            x = [train[0][t] for t in train_index]

            z = np.array([img_feats[:, train[1][t]] for t in train_index])

            x, mask, y = prepare_data(x)

            if (x.shape[0] == batch_size):
                feed_dict = {image_fea: z, input_seqs: x, target_seqs: y, input_mask: mask}
                _, loss_value = sess.run([train_op, total_loss], feed_dict=feed_dict)
                if steps % 1000 == 0:  # print loss every 1000 steps
                    print("steps:" + str(steps + eidx * 17710))
                    print("loss_value:" + str(loss_value))


if __name__ == "__main__":
    tf.app.run()
