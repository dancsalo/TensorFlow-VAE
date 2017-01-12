#!/usr/bin/env python

"""
@author: Dan Salo, Jan 2017

Purpose: Implement Convolutional Variational Autoencoder for Semi-Supervision with partially-labeled MNIST dataset.
MNIST Dataset will be downloaded and batched automatically.

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
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'image_dim': 28,
    'hidden_size': 10,
    'num_classes': 10,
    'batch_size': 100,
    'display_step': 200,
    'weight_decay': 1e-6,
    'learning_rate': 0.001,
    'epochs': 100,
    'run_num': 3,
}


class ConvVae(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)

    def _data(self):
        """ Define data I/O """
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon'])
        self.data = Mnist(self.flags)

    def _summaries(self):
        """ Define summaries for Tensorboard """
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Reconstruction_Loss", self.recon)
        tf.summary.scalar("VAE_Loss", self.vae)
        tf.summary.scalar("Weight_Decay_Loss", self.weight)
        tf.summary.histogram("Mean", self.mean)
        tf.summary.histogram("Stddev", self.stddev)
        tf.summary.image("x", self.x)
        tf.summary.image("x_hat", self.x_hat)

    def _encoder(self, x):
        """Define q(z|x) network"""
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
        """ Define p(x|z) network"""
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
        """ Define network """
        with tf.variable_scope("model"):
            self.latent = self._encoder(x=self.x)
            self.x_hat, self.mean, self.stddev = self._decoder(z=self.latent)
        with tf.variable_scope("model", reuse=True):
            self.x_gen, _, _ = self._decoder(z=None)

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        epsilon = 1e-8
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.recon = const * tf.reduce_sum(tf.squared_difference(self.x, self.x_hat))
        self.vae = const * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.flags['learning_rate']).minimize(self.cost)

    def _generate_train_batch(self):
        """ Generate a training batch of images """
        self.train_batch_y, self.train_batch_x = self.data.next_train_batch(self.flags['batch_size'])
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])

    def _run_train_iter(self):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer],
                                        feed_dict={self.x: self.train_batch_x, self.epsilon: self.norm})
        return summary

    def _run_train_metrics_iter(self):
        """ Run training iteration and also calculate metrics """
        summary, self.loss, self.x_recon, _ =\
            self.sess.run([self.merged, self.cost, self.x_hat, self.optimizer],
                          feed_dict={self.x: self.train_batch_x, self.epsilon: self.norm})
        return summary

    def _record_train_metrics(self):
        """ Record training metrics """
        for j in range(1):
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '.png',
                              np.squeeze(self.train_batch_x[j]))
            scipy.misc.imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '.png',
                              np.squeeze(self.x_recon[j]))
        self.print_log("Batch Number: " + str(self.step) + ", Image Loss= " + "{:.6f}".format(self.loss))

    def train(self):
        """ Train the autoencoder """
        for i in range(self.flags['epochs'] * self.data.num_training_images):
            self.print_log('Learning Rate: %d' % self.learn_rate)
            self.print_log('Iterations: %d' % self.iters_num)
            while self.step < self.iters_num:
                print('Batch number: %d' % self.step)
                self._generate_train_batch()
                if self.step % self.flags['display_step'] != 0:
                    summary = self._run_train_iter()
                else:
                    summary = self._run_train_metrics_iter()
                    self._record_train_metrics()
                self._record_training_step(summary)
            self._save_model(section=i)


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_vae = ConvVae(flags, run_num=flags['run_num'])
    model_vae.train()

if __name__ == "__main__":
    main()
