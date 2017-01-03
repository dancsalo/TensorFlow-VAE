#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Implement Convolutional VAE for MNIST dataset to demonstrate NNClasses functionality
"""

import sys
sys.path.append('../')

from TensorBase.tensorbase.base import Model
from TensorBase.tensorbase.base import Layers

import tensorflow as tf
import numpy as np
import scipy.misc
import time


# Global Dictionary of Flags
flags = {
    'save_directory': 'summaries/',
    'model_directory': 'conv_vae_semi/',
    'train_data_file': 'mnist_1000_train.tfrecords',
    'valid_data_file': 'mnist_1000_valid.tfrecords',
    'test_data_file': 'mnist_1000_test.tfrecords',
    'restore': False,
    'restore_file': 'part_1.ckpt.meta',
    'image_dim': 28,
    'hidden_size': 64,
    'num_classes': 10,
    'batch_size': 100,
    'xentropy': 10,
    'display_step': 200,
    'starter_lr': 1e-3,
    'num_epochs': 75,
    'weight_decay': 1e-6,
}


class ConvVae(Model):
    def __init__(self, flags_input, run_num, labeled):
        for n in ['train', 'valid', 'test']:
            if n == 'train':
                flags_input[n + '_data_file'] = 'data/mnist_' +str(labeled) + '_' + n + '_labeled.tfrecords'
            else:
                flags_input[n + '_data_file'] = 'data/mnist_' +str(labeled) + '_' + n + '.tfrecords'
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        self.print_log('Number of Labeled: %d' % int(labeled))
        self.flags['xentropy'] = 0.1 * int(labeled)

    def _set_placeholders(self):
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.train_x, self.train_y = self.batch_inputs("train")
        self.num_train_images = 55000
        self.num_valid_images = 5000
        self.num_test_images = 10000

    def _set_summaries(self):
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Reconstruction_Loss", self.recon)
        tf.summary.scalar("VAE_Loss", self.vae)
        tf.summary.scalar("XEntropy_Loss", self.xentropy)
        tf.summary.histogram("Mean", self.mean)
        tf.summary.histogram("Stddev", self.stddev)
        tf.summary.image("train_x", self.train_x)
        tf.summary.image("x_hat", self.x_hat)

    def _encoder(self, x):
        encoder = Layers(x)
        encoder.conv2d(5, 32)
        encoder.maxpool()
        encoder.conv2d(5, 64, stride=2)
        encoder.conv2d(7, 128, padding='VALID')
        encoder.conv2d(1, self.flags['hidden_size'] * 2, activation_fn=None)
        return encoder.get_output()

    def _decoder(self, z):
        if z is None:
            mean = None
            stddev = None
            logits = None
            input_sample = self.epsilon
        else:
            z = tf.reshape(z, [-1, self.flags['hidden_size'] * 2])
            print(z.get_shape())
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            mlp = Layers(mean)
            mlp.fc(self.flags['num_classes'])
            logits = tf.nn.softmax(mlp.get_output())
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(3, 128, padding='VALID')
        decoder.deconv2d(3, 64, padding='VALID', stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(5, 32, stride=2)
        decoder.deconv2d(7, 1, activation_fn=tf.nn.tanh, s_value=None)
        return decoder.get_output(), mean, stddev, logits

    def _network(self):
        with tf.variable_scope("model") as scope:
            self.latent = self._encoder(x=self.train_x)
            self.x_hat, self.mean, self.stddev, self.logits_train = self._decoder(z=self.latent)     

    def _optimizer(self):
        epsilon = 1e-8
        self.learning_rate = self.flags['starter_lr']
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.xentropy = const * self.flags['xentropy'] * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.logits_train, self.train_y, name='xentropy'))
        self.recon = const * tf.reduce_sum(tf.squared_difference(self.train_x, self.x_hat))
        self.vae = const * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        #self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.xentropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def _run_train_iter(self):
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        self.summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.epsilon: self.norm})

    def _run_train_summary_iter(self):
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        self.summary, self.loss, self.x_recon, self.x_true, _ = self.sess.run([self.merged, self.cost, self.x_hat, self.train_x, self.optimizer], feed_dict={self.epsilon: self.norm})

    def run(self, mode):
        self.step = 0
        if mode != "train":
            self.sess.close()
            tf.reset_default_graph()
            self.results = list()
            self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
            self.flags['restore'] = True
            self.flags['restore_file'] = 'part_1.ckpt.meta'
            self.eval_x, self.eval_y = self.batch_inputs(mode)
            norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
            with tf.variable_scope("model") as scope:
                self.latent = self._encoder(x=self.eval_x)
                _, _, _, self.logits_eval = self._decoder(z=self.latent)
            _, _, self.sess, _ = self._set_tf_functions()
            self._initialize_model()
        coord = tf.train.Coordinator()
        try:
            threads = list()
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(self.sess, coord=coord, daemon=True,start=True))
            while not coord.should_stop():
                start_time = time.time()
                self.duration = time.time() - start_time
                if mode == "train":
                    self._run_train_iter()
                    self._record_training_step()
                    if self.step % self.flags['display_step'] == 0:
                        self._run_train_summary_iter()
                        self._record_train_metrics()
                else:
                    logits, true = self.sess.run([self.logits_eval, self.eval_y], feed_dict={self.epsilon: norm})
                    # predictions = np.reshape(logits, [-1, self.flags['num_classes']])
                    correct_prediction = np.equal(np.argmax(true, 1), np.argmax(logits, 1))
                    self.results = np.concatenate((self.results, correct_prediction))
                    self.step += 1
                print(self.step)
        except Exception as e:
            coord.request_stop(e)
        finally:
            if mode == "train":
                self._save_model(section=1)
            else:  # eval mode
                self._record_eval_metrics(mode)
            self.print_log('Finished ' + mode + ': %d epochs, %d steps.' % (self.flags['num_epochs'], self.step))
        coord.request_stop()  
        coord.join(threads, stop_grace_period_secs=10)
    
    def close(self):
        self.sess.close()
        tf.reset_default_graph()
    
    def _record_train_metrics(self):
        for j in range(1):
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '_' + str(j) +'.png',
                              np.squeeze(self.x_true[j]))
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '_' + str(j) + '.png',
                              np.squeeze(self.x_recon[j]))
        print(self.x_true[0].mean())
        print(self.x_recon[0].mean())
        print(self.x_true[0].min())
        print(self.x_recon[0].min())
        print(self.x_true[0].max())
        print(self.x_recon[0].max())
        self.print_log('Step %d: loss = %.6f (%.3f sec)' % (self.step,self.loss,self.duration))

    def _record_eval_metrics(self, mode):
        accuracy = np.mean(self.results)
        self.print_log("Accuracy on "+ mode + " Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'ValidAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()
    
    def batch_inputs(self, dataset):
        with tf.name_scope('batch_processing'):
            # Approximate number of examples per shard.
            examples_per_shard = 1024
            min_queue_examples = examples_per_shard * 16
            if dataset == "train":
                filename = self.flags['train_data_file']
                epochs = self.flags['num_epochs']
            elif dataset == "valid":
                filename = self.flags['valid_data_file']
                epochs = 1
            else:  # test data file
                filename = self.flags['test_data_file']
                epochs = 1
            filename_queue = tf.train.string_input_producer([filename],
                num_epochs=epochs,shuffle=True,capacity=16)
            examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 *  self.flags['batch_size'],
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
            num_preprocess_threads=4
            num_readers=4
            enqueue_ops = list()
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
            images_and_labels = list()
            for _ in range(num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image, label = self.read_and_decode(example_serialized)
                images_and_labels.append([image, label])
            image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=self.flags['batch_size'], capacity=2 * num_preprocess_threads * self.flags['batch_size'])
            return image_batch, label_batch

    def read_and_decode(self, example_serialized):
        features = tf.parse_single_example(
            example_serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([self.flags['num_classes']], tf.int64, default_value=[-1]*self.flags['num_classes']),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
            })
        # now return the converted data
        label = features['label']
        image = tf.decode_raw(features['image'], tf.float32)
        image.set_shape([784])
        image = tf.reshape(image, [28, 28, 1])
        image = (image - 0.5) * 2  # max value = 1, min value = -1
        return image, tf.cast(label, tf.int32)

def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    run_num = sys.argv[1]
    labels = sys.argv[2]
    model = ConvVae(flags, run_num=run_num, labeled=labels)
    model.run("train")
    model.run("valid")
    model.run("test")

if __name__ == "__main__":
    main()
