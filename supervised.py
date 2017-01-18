#!/usr/bin/env python

"""
@author: Dan Salo, Jan 2017

Purpose: Implement Convolutional Variational Autoencoder for Classification of Fully-Labeled MNIST Dataset
Use mnist_process.py to generate training, validation and test files.

"""

from tensorbase.base import Data, Model, Layers

import sys
import tensorflow as tf
import numpy as np
import math


# Global Dictionary of Flags
flags = {
    'data_directory': 'MNIST_data/',
    'save_directory': 'summaries/',
    'model_directory': 'conv/',
    'train_data_file': 'mnist_1_train.tfrecords',
    'valid_data_file': 'data/mnist_valid.tfrecords',
    'test_data_file': 'data/mnist_test.tfrecords',
    'restore': False,
    'restore_file': 'part_1.ckpt.meta',
    'image_dim': 28,
    'num_classes': 10,
    'batch_size': 100,
    'display_step': 250,
    'starter_lr': 1e-3,
    'num_epochs': 100,
}


class Conv(Model):
    def __init__(self, flags_input, run_num, labeled):
        flags_input['train_data_file'] = 'data/mnist_' + str(labeled) + '_train_labeled.tfrecords'
        super().__init__(flags_input, run_num)
        self.print_log('Number of Labeled: %d' % int(labeled))

    def eval_model_init(self):
        """ Initialize model for evaluation """
        self.sess.close()
        tf.reset_default_graph()
        self.results = list()
        self.flags['restore'] = True
        self.flags['restore_file'] = 'part_1.ckpt.meta'
        file = self.flags['test_data_file']
        self.eval_x, self.eval_y = Data.batch_inputs(self.read_and_decode, file, self.flags['batch_size'], mode="eval")
        with tf.variable_scope("model"):
            _, self.logits_eval = self._encoder(x=self.eval_x)
        self.sess = self._define_sess()
        self._initialize_model()

    def _data(self):
        """ Define data and batching """
        file = self.flags['train_data_file']
        self.train_x, self.train_y = Data.batch_inputs(self.read_and_decode, file, self.flags['batch_size'])
        self.num_train_images = 55000
        self.num_valid_images = 5000
        self.num_test_images = 10000

    def _summaries(self):
        """ Define summaries for Tensorboard """
        tf.summary.scalar("XEntropy_Loss", self.cost)
        
    def _encoder(self, x):
        """ Define CNN for classification """
        encoder = Layers(x)
        encoder.conv2d(5, 32)
        encoder.maxpool()
        encoder.conv2d(5, 64, stride=2)
        encoder.conv2d(7, 128, padding='VALID')
        encoder.conv2d(1, 64, activation_fn=None)
        encoder.flatten()
        encoder.fc(self.flags['num_classes'], activation_fn=None)
        logits = tf.nn.softmax(encoder.get_output())
        return encoder.get_output(), logits

    def _network(self):
        """ Define network """
        with tf.variable_scope("model"):
            self.y_hat, self.logits_train = self._encoder(x=self.train_x)

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        self.learning_rate = self.flags['starter_lr']
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.cost = const * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_y, logits=self.y_hat, name='xentropy'))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def _run_train_iter(self):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer])
        return summary

    def _run_train_metrics_iter(self):
        """ Run training iteration with metrics output """
        summary, self.loss, _ = self.sess.run([self.merged, self.cost, self.optimizer])
        return summary

    def run(self, mode):
        """ Run either train function or eval function """
        if mode != "train":
            self.eval_model_init()
            threads, coord = Data.init_threads(self.sess)
            self.eval(coord, mode)
        else:
            threads, coord = Data.init_threads(self.sess)
            self.train()
        self.print_log('Finished ' + mode + ': %d epochs, %d steps.' % (self.flags['num_epochs'], self.step))
        Data.exit_threads(threads, coord)

    def train(self):
        """ Run training function. Save model upon completion """
        iterations = math.ceil(self.num_train_images/self.flags['batch_size']) * self.flags['num_epochs']
        self.print_log('Training for %d iterations' % iterations)
        for i in range(iterations):
            if self.step % self.flags['display_step'] != 0:
                summary = self._run_train_iter()
            else:
                summary = self._run_train_metrics_iter()
                self._record_train_metrics()
            self._record_training_step(summary)
            print(self.step)
        self._save_model(section=1)

    def eval(self, coord, mode):
        """ Run evaluation function. Save accuracy or other metrics upon completion """
        try:
            while not coord.should_stop():
                logits, true = self.sess.run([self.logits_eval, self.eval_y], feed_dict={self.epsilon: norm})
                correct_prediction = np.equal(np.argmax(true, 1), np.argmax(logits, 1))
                self.results = np.concatenate((self.results, correct_prediction))
                self.step += 1
                print(self.step)
        except Exception as e:
            coord.request_stop(e)
        finally:
            self._record_eval_metrics(mode)

    def _record_train_metrics(self):
        """ Record training metrics """
        self.print_log('Step %d: loss = %.6f' % (self.step, self.loss))

    def _record_eval_metrics(self, mode):
        """ Record evaluation metrics """
        accuracy = np.mean(self.results)
        self.print_log("Accuracy on " + mode + " Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + mode + '_Accuracy.txt', 'w')
        file.write(mode + 'set accuracy:')
        file.write(str(accuracy))
        file.close()

    def read_and_decode(self, example_serialized):
        """ Read and decode binarized, raw MNIST dataset from .tfrecords file generated by MNIST.py """
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
    model = Conv(flags, run_num=run_num, labeled=labels)
    model.run("train")
    model.run("valid")
    model.run("test")

if __name__ == "__main__":
    main()
