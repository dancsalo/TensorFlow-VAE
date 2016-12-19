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
from Datasets.Weapons import Weapons

import tensorflow as tf
import numpy as np
import scipy.misc
import pickle


# Global Dictionary of Flags
flags = {
    'data_directory': '/media/kd/9ef888dc-a92d-4587-af71-bb562dbc5764/justWeapons/',
    'save_directory': 'summaries/',
    'model_directory': 'conv_vae_bags/',
    'restore': False,
    'restore_file': 'part_0.ckpt.meta',
    'datasets': 'Bags',
    'image_dim': 128,
    'hidden_size': 128,
    'batch_size': 64,
    'display_step': 500,
    'weight_decay': 1e-8,
    'lr_decay': 0.95,
    'vae_increase': 0.000001,
    'run_num': 9,
    'lr_iters': [(1e-3, 7500), (1e-4, 7500), (1e-5, 10000), (1e-6, 15000), (1e-7, 20000)]
}


class ConvVae(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        with open(flags['data_directory'] + 'image_dict.pickle', 'rb') as pickle_file:
            image_dict = pickle.load(pickle_file)
        self.data = Weapons(image_dict, flags_input)
        self.vae_value = 0

    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.vae_const = tf.placeholder(tf.float32, name='vae_const')

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
        encoder.conv2d(3, 256, stride=2)
        encoder.conv2d(3, 256)
        encoder.conv2d(3, 512, stride=2)
        encoder.conv2d(3, 512)
        encoder.conv2d(3, 1024, stride=2)
        encoder.conv2d(3, 1024)
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
        decoder.deconv2d(4, 1024, padding='VALID')
        decoder.deconv2d(3, 1024)
        decoder.deconv2d(3, 512, stride=2)
        decoder.deconv2d(3, 512)
        decoder.deconv2d(3, 256, stride=2)
        decoder.deconv2d(3, 256)
        decoder.deconv2d(3, 128, stride=2)
        decoder.deconv2d(3, 128)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(3, 64)
        decoder.deconv2d(5, 1, stride=2, activation_fn=tf.nn.tanh)
        return decoder.get_output(), mean, stddev

    def _network(self):
        with tf.variable_scope("model"):
            self.latent = self._encoder(x=self.x)
            self.x_hat, self.mean, self.stddev = self._decoder(z=self.latent)
        with tf.variable_scope("model", reuse=True):
            self.x_gen, _, _ = self._decoder(z=None)

    def _optimizer(self):
        epsilon = 1e-8
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.recon = const * tf.reduce_sum(tf.squared_difference(self.x, self.x_hat))
        self.vae = const * self.vae_const * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def _generate_train_batch(self):
        _, self.train_batch_x = self.data.next_train_batch(self.flags['batch_size'])
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])

    def _run_train_iter(self):
        if self.vae_value < 1:
            self.vae_value += self.flags['vae_increase']
        self.summary, _ = self.sess.run([self.merged, self.optimizer],
                                   feed_dict={self.x: self.train_batch_x, self.epsilon: self.norm,
                                              self.lr: self.learn_rate, self.vae_const: self.vae_value})

    def _run_train_summary_iter(self):
        self.learn_rate = self.learn_rate * self.flags['lr_decay']
        self.print_log('Learning Rate: %f ' % self.learn_rate)
        if self.vae_value < 1:
            self.vae_value += self.flags['vae_increase']
        norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        self.summary, self.loss, self.x_recon, _ =\
            self.sess.run([self.merged, self.cost, self.x_hat, self.optimizer],
                          feed_dict={self.x: self.train_batch_x, self.epsilon: norm,self.lr: self.learn_rate, self.vae_const: self.vae_value})

    def _record_train_metrics(self):
        for j in range(1):
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '.png',
                              np.squeeze(self.train_batch_x[j,:,:,0]))
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '.png',
                              np.squeeze(self.x_recon[j,:,:,0]))
        self.print_log("Batch Number: " + str(self.step) + ", Image Loss= " + "{:.6f}".format(self.loss))


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_vae = ConvVae(flags, run_num=flags['run_num'])
    model_vae.train()

if __name__ == "__main__":
    main()
