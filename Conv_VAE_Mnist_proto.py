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
    'num_labels': 1000,
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'image_dim': 28,
    'hidden_size': 64,
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
        self.data = Mnist(flags_input)
        tf.train.start_queue_runners(sess=self.sess)

    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _set_summaries(self):
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Reconstruction Loss", self.recon)
        tf.scalar_summary("VAE Loss", self.vae)
        tf.scalar_summary("Weight Decay Loss", self.weight)
        tf.histogram_summary("Mean", self.mean)
        tf.histogram_summary("Stddev", self.stddev)
        tf.image_summary("x", self.x)
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
            input_sample = self.epsilon
        else:
            z = tf.reshape(z, [-1, self.flags['hidden_size'] * 2])
            print(z.get_shape())
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(3, 128, padding='VALID')
        decoder.deconv2d(3, 128, padding='VALID', stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(5, 1, activation_fn=tf.nn.tanh, s_value=None)
        return decoder.get_output(), mean, stddev

    def _network(self):
        with tf.variable_scope("model"):
            self.latent = self._encoder(x=self.x)
            self.x_hat, self.mean, self.stddev = self._decoder(z=self.latent)
        with tf.variable_scope("model", reuse=True):
            self.x_gen, _, _ = self._decoder(z=None)

    def _optimizer(self):
        epsilon = 1e-8
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(flags['starter_lr'], global_step, 100000, 0.96, staircase=True)
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.recon = const * tf.reduce_sum(tf.squared_difference(self.x, self.x_hat))
        self.vae = const * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=self.global_step)

    def _generate_train_batch(self):
        label, image = self.read_and_decode_single_example(self.flags['data_file'])
        image = tf.cast(image, tf.float32) / 255.
        self.train_batch_y, self.train_batch_x = tf.train.shuffle_batch([label, image],
                                                                        capacity=2000,
                                                                        batch_size=self.flags['batch_size'],
                                                                        min_after_dequeue=1000)
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])

    def _run_train_iter(self):
        self.summary, _ = self.sess.run([self.merged, self.optimizer])

    def _run_train_summary_iter(self):
        self.summary, self.loss, self.x_recon, _ = self.sess.run([self.merged, self.cost, self.x_hat, self.optimizer])

    def _record_train_metrics(self):
        for j in range(1):
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '.png',
                              np.squeeze(self.train_batch_x[j]))
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '.png',
                              np.squeeze(self.x_recon[j]))
        self.print_log("Batch Number: " + str(self.step) + ", Image Loss= " + "{:.6f}".format(self.loss))

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
        return label, image


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_vae = ConvVae(flags, run_num=flags['run_num'])
    model_vae.train()

if __name__ == "__main__":
    main()
