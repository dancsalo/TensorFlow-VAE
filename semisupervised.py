#!/usr/bin/env python

"""
@author: Dan Salo, Jan 2017

Purpose: Implement Convolutional Variational Autoencoder for Semi-Supervision with partially-labeled MNIST dataset.
Use mnist_process.py to generate training, validation and test files.

"""

from tensorbase.base import Data, Model, Layers
from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math


# Global Dictionary of Flags
flags = {
    'save_directory': 'summaries/',
    'model_directory': 'conv_vae_semi/',
    'train_data_file': 'data/mnist_1000_train.tfrecords',
    'valid_data_file': 'data/mnist_valid.tfrecords',
    'test_data_file': 'data/mnist_test.tfrecords',
    'restore': False,
    'restore_file': 'part_1.ckpt.meta',
    'image_dim': 28,
    'hidden_size': 64,
    'num_classes': 10,
    'batch_size': 100,
    'display_step': 550,
    'starter_lr': 1e-4,
    'num_epochs': 75,
    'weight_decay': 1e-6,
    'run_num': 0,
}


class ConvVaeSemi(Model):
    def __init__(self, flags_input, run_num, labeled):
        """ Define the labeled and unlabeled file names. Use queueing and threading I/O. Initialize Model.init()"""
        flags_input['train_unlabeled_data_file'] = 'data/mnist_' + str(labeled) + '_train_unlabeled.tfrecords'
        flags_input['train_labeled_data_file'] = 'data/mnist_' + str(labeled) + '_train_labeled.tfrecords'
        super().__init__(flags_input, run_num)
        self.labeled = int(labeled)
        self.print_log('Number of Labeled: %d' % int(labeled))

    def eval_model_init(self):
        self.sess.close()
        tf.reset_default_graph()
        self.step = 1
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.flags['restore'] = True
        self.flags['restore_file'] = 'part_1.ckpt.meta'
        self.eval_x, self.eval_y = Data.batch_inputs(mode)
        with tf.variable_scope("model"):
            self.latent = self._encoder(x=self.eval_x)
            _, _, _, _, self.logits_eval = self._decoder(z=self.latent)
        _, _, self.sess, _ = self._set_tf_functions()
        self._initialize_model()

    def _data(self):
        """Define data I/O"""
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.num_train_images = 55000
        self.num_valid_images = 5000
        self.num_test_images = 10000
        # Load in training data of batch_size/2, and combine into train_x, train_y of size batch_size
        file = self.flags['train_unlabeled_data_file']
        unlabeled_x, unlabeled_y = Data.batch_inputs(self.read_and_decode, file, int(self.flags['batch_size']/2))
        file = self.flags['train_labeled_data_file']
        labeled_x, labeled_y = Data.batch_inputs(self.read_and_decode, file, int(self.flags['batch_size']/2))
        self.train_x = tf.concat(0, [labeled_x, unlabeled_x])
        self.train_y = tf.concat(0, [labeled_y, unlabeled_y])

    def _summaries(self):
        """Define summaries for tensorboard"""
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Reconstruction_Loss", self.recon)
        tf.summary.scalar("VAE_Loss", self.vae)
        tf.summary.scalar("XEntropy_Loss", self.xentropy)
        tf.summary.histogram("Mean", self.mean)
        tf.summary.histogram("Stddev", self.stddev)
        tf.summary.image("train_x", self.train_x)
        tf.summary.image("x_hat", self.x_hat)

    def _encoder(self, x):
        """Define q(z|x) network"""
        encoder = Layers(x)
        encoder.conv2d(5, 32)
        encoder.maxpool()
        encoder.conv2d(5, 64, stride=2)
        encoder.conv2d(7, 128, padding='VALID')
        encoder.conv2d(1, self.flags['hidden_size'] * 2, activation_fn=None)
        return encoder.get_output()

    def _decoder(self, z):
        """Define p(x|z) network"""
        if z is None:
            mean = None
            stddev = None
            logits = None
            class_predictions = None
            input_sample = self.epsilon
        else:
            z = tf.reshape(z, [-1, self.flags['hidden_size'] * 2])
            mean, stddev = tf.split(1, 2, z)  # Compute latent variables (z) by calculating mean, stddev
            stddev = tf.sqrt(tf.exp(stddev))
            mlp = Layers(mean)
            mlp.fc(self.flags['num_classes'])
            class_predictions = mlp.get_output()
            logits = tf.nn.softmax(class_predictions)
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(3, 128, padding='VALID')
        decoder.deconv2d(3, 64, padding='VALID', stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(5, 32, stride=2)
        decoder.deconv2d(7, 1, activation_fn=tf.nn.tanh, s_value=None)
        return decoder.get_output(), mean, stddev, class_predictions, logits

    def _network(self):
        """ Define network outputs """
        with tf.variable_scope("model"):
            self.latent = self._encoder(x=self.train_x)
            self.x_hat, self.mean, self.stddev, preds, logits_train = self._decoder(z=self.latent)
            self.preds = preds[0:int(self.flags['batch_size']/2), ]
            self.logits_train = logits_train[0:int(self.flags['batch_size']/2), ]
            self.train_y_labeled = self.train_y[0:int(self.flags['batch_size']/2)]

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        epsilon = 1e-8
        self.learning_rate = self.flags['starter_lr']
        const_vae = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.xentropy = 2/(self.flags['batch_size']) * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.preds, self.train_y_labeled, name='xentropy'))
        self.recon = const_vae * tf.reduce_sum(tf.squared_difference(self.train_x, self.x_hat))
        self.vae = const_vae * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.xentropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def _run_train_iter(self):
        """ Run training iteration """
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.epsilon: self.norm})
        return summary

    def _run_train_metrics_iter(self):
        """ Run training iteration with metrics update """
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
        summary, self.loss, self.x_recon, self.x_true, logits, true_y, _ = self.sess.run([self.merged, self.cost, self.x_hat, self.train_x, self.logits_train, self.train_y_labeled, self.optimizer], feed_dict={self.epsilon: self.norm})
        correct_prediction = np.equal(np.argmax(true_y, 1), np.argmax(logits, 1))
        self.print_log('Training Minibatch Accuracy: %.6f' % np.mean(correct_prediction))
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
        norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
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
        """ Record training metrics and print to log and terminal """
        for j in range(1):
            imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '_' + str(j) + '.png', np.squeeze(self.x_true[j]))
            imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '_' + str(j) + '.png', np.squeeze(self.x_recon[j]))
        self.print_log('Step %d: loss = %.6f' % (self.step, self.loss))

    def _record_eval_metrics(self, mode):
        """ Record evaluation metrics and print to log and terminal """
        accuracy = np.mean(self.results)
        self.print_log("Accuracy on " + mode + " Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + mode + '_Accuracy.txt', 'w')
        file.write(mode + 'set accuracy:')
        file.write(str(accuracy))
        file.close()

    def read_and_decode(self, example_serialized):
        """ Read and decode binarized, raw MNIST dataset from .tfrecords file generated by MNIST.py """
        num = self.flags['num_classes']

        # Parse features from binary file
        features = tf.parse_single_example(
            example_serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([num], tf.int64, default_value=[-1] * num),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
            })
        # Return the converted data
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
    model = ConvVaeSemi(flags, run_num=run_num, labeled=labels)
    model.run("train")
    model.run("valid")
    model.run("test")

if __name__ == "__main__":
    main()
