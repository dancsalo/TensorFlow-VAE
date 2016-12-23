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
from TensorBase.tensorbase.data import Mnist

import tensorflow as tf
import numpy as np
import scipy.misc


# Global Dictionary of Flags
flags = {
    'data_directory': 'MNIST_data/',
    'save_directory': 'summaries/',
    'model_directory': 'conv_vae/',
    'train_data_file': 'mnist_1000_train.tfrecords',
    'valid_data_file': 'mnist_1000_valid.tfrecords',
    'test_data_file': 'mnist_1000_test.tfrecords',
    'num_labels': 1000,
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'image_dim': 28,
    'hidden_size': 5,
    'num_classes': 10,
    'batch_size': 128,
    'display_step': 500,
    'starter_lr': 0.001,
    'weight_decay': 1e-4,
    'lr_iters': [(1e-3, 10000), (1e-4, 10000), (1e-5, 10000)],
    'run_num': 1,
}


class ConvVae(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        tf.train.start_queue_runners(sess=self.sess)

    def _set_placeholders(self):
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        label_tr, image_tr = self.read_and_decode_single_example(self.flags['train_data_file'])
        image_tr = tf.cast(image_tr, tf.float32) / 255.
        self.train_y, train_x = tf.train.shuffle_batch([label_tr, image_tr], capacity=2000, batch_size=self.flags['batch_size'],
                                                  min_after_dequeue=1000)
        self.train_x = self.img_norm(train_x)
        label_v, image_v = self.read_and_decode_single_example(self.flags['valid_data_file'])
        image_v = tf.cast(image_v, tf.float32) / 255.
        self.valid_y, valid_x = tf.train.shuffle_batch([label_v, image_v], capacity=2000, batch_size=self.flags['batch_size'],
                                                  min_after_dequeue=1000)
        self.valid_x = self.img_norm(valid_x)
        label_t, image_t = self.read_and_decode_single_example(self.flags['test_data_file'])
        image_t = tf.cast(image_t, tf.float32) / 255.
        self.test_y, test_x = tf.train.shuffle_batch([label_t, image_t], capacity=2000, batch_size=self.flags['batch_size'],
                                                  min_after_dequeue=1000)
        self.test_x = self.img_norm(test_x)
        self.num_train_images = 55000
        self.num_valid_images = 5000
        self.num_test_images = 10000

    @staticmethod
    def img_norm(x, epsilon=1e-6):
        """
        :param x: input feature map stack
        :param scope: name of tensorflow scope
        :param epsilon: float
        :return: output feature map stack
        """
        # Calculate batch mean and variance
        mean, var = tf.nn.moments(x,axes=[0,1,2], keep_dims=True)
        out = (x -mean) / tf.sqrt(var + epsilon)
        return tf.expand_dims(out, 3)

    def _set_summaries(self):
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Reconstruction Loss", self.recon)
        tf.scalar_summary("VAE Loss", self.vae)
        tf.scalar_summary("Weight Decay Loss", self.weight)
        tf.histogram_summary("Mean", self.mean)
        tf.histogram_summary("Stddev", self.stddev)
        tf.image_summary("train_x", self.train_x)
        tf.image_summary("x_hat", self.x_hat)

    def _encoder(self, x):
        encoder = Layers(x)
        encoder.conv2d(5, 64)
        encoder.maxpool()
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 128, stride=2)
        encoder.conv2d(3, 128)
        encoder.conv2d(1, 64)
        encoder.conv2d(1, self.flags['hidden_size'] * 2, activation_fn=None)
        encoder.avgpool(globe=True)
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
            logits = tf.nn.softmax(mean)
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(3, 128, padding='VALID')
        decoder.deconv2d(3, 128, padding='VALID', stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(5, 1, activation_fn=tf.nn.tanh, s_value=None)
        return decoder.get_output(), mean, stddev, logits

    def _network(self):
        with tf.variable_scope("model"):
            self.latent = self._encoder(x=self.train_x)
            self.x_hat, self.mean, self.stddev, _ = self._decoder(z=self.latent)
        with tf.variable_scope("model", reuse=True):
            latent_valid = self._encoder(x=self.valid_x)
            _, _, _, self.logits_valid = self._decoder(z=latent_valid)
        with tf.variable_scope("model", reuse=True):
            latent_test = self._encoder(x=self.test_x)
            _, _, _, self.logits_test = self._decoder(z=latent_test)
            self.x_gen, _, _, _ = self._decoder(z=None)

    @staticmethod
    def split_y(train_y, true_y):
        inds = list()
        for ind in range(train_y.get_shape()[0]):
            if np.sum(train_y[ind]) == 1:
                inds.append(ind)
        return train_y[inds], true_y[inds]

    def _optimizer(self):
        epsilon = 1e-8
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.flags['starter_lr'], global_step, 100000, 0.96, staircase=True)
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        train_y, true_y = self.split_y(tf.reshape(self.latent, [-1, self.flags['num_classes']]), self.train_y)
        self.xentropy = const * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(true_y, train_y, name='xentropy'))
        self.recon = const * tf.reduce_sum(tf.squared_difference(self.train_x, self.x_hat))
        self.vae = const * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=global_step)

    def _run_train_iter(self):
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        self.summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.epsilon: self.norm})

    def _run_train_summary_iter(self):
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        self.summary, self.loss, self.x_recon, _ = self.sess.run([self.merged, self.cost, self.x_hat, self.optimizer], feed_dict={self.epsilon: self.norm})

    def _run_valid_iter(self):
        logits = self.sess.run([self.logits_valid], feed_dict={self.epsilon: self.norm})
        predictions = np.reshape(logits, [-1, self.flags['num_classes']])
        correct_prediction = np.equal(np.argmax(self.valid_y, 1), np.argmax(predictions, 1))
        self.valid_results = np.concatenate((self.valid_results, correct_prediction))

    def _run_test_iter(self):
        logits = self.sess.run([self.logits_test], feed_dict={self.epsilon: self.norm})
        predictions = np.reshape(logits, [-1, self.flags['num_classes']])
        correct_prediction = np.equal(np.argmax(self.test_y, 1), np.argmax(predictions, 1))
        self.test_results = np.concatenate((self.test_results, correct_prediction))

    def _record_train_metrics(self):
        for j in range(1):
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '.png',
                              np.squeeze(self.train_batch_x[j]))
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '.png',
                              np.squeeze(self.x_recon[j]))
        self.print_log("Batch Number: " + str(self.step) + ", Image Loss= " + "{:.6f}".format(self.loss))

    def _record_valid_metrics(self):
        accuracy = np.mean(self.valid_results)
        self.print_log("Accuracy on Validation Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'ValidAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()

    def _record_test_metrics(self):
        accuracy = np.mean(self.test_results)
        self.print_log("Accuracy on Test Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'TestAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()

    @staticmethod
    def read_and_decode_single_example(filename):
        filename_queue = tf.train.string_input_producer([filename],num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([10], tf.int64),
                'image': tf.FixedLenFeature([28, 28], tf.int64)
            })
        # now return the converted data
        label = features['label']
        image = features['image']
        return tf.cast(label, tf.float32), image


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_vae = ConvVae(flags, run_num=flags['run_num'])
    model_vae.train()

if __name__ == "__main__":
    main()
